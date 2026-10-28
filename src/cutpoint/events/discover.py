"""Game-agnostic HUD region discovery for the OCR layer.

``cutpoint events --discover-regions`` (or automatically when a game has no preset)
samples short bursts of frames spread across the game's sources, runs
full-frame RapidOCR on each and finds where text *repeatedly appears and
changes*:

* Frames are taken in bursts (default 40 bursts x 4 frames, 1.5 s apart), so
  "changes over time" can be measured where it matters: within seconds, as a
  kill feed or a banner does, not across minutes.
* Menu / scoreboard frames are dropped (:func:`gameplay_frames`): text spread
  over most of the screen centre, or far more text boxes than the source
  usually shows. A busy HUD (many boxes along the edges, in every frame) stays.
* Every text box is rasterised onto a coarse grid (1920x1080 reference, 24 px
  cells). Per cell: how often text is there (occupancy) and how often its text
  differs between consecutive frames of a burst (change rate). Cells with
  frequent static text (ammo labels, minimap letters) are ignored.
* Connected groups of "dynamic" cells become regions, classified from the
  boxes inside them:
    - ``feed``: stacked lines (2+ rows per frame) near a screen edge or corner,
      fewer than :data:`FEED_SCORE_ROWS` of them a "+" number alone, grown up
      and down by rows that continue it at its row pitch but show too rarely to be
      dynamic (:func:`_grow_feed`, up to twice its height);
        - icon feeds: such a region whose lines are often icon kill lines (two names
          split by an icon gap, :func:`_icon_lines`) is a feed even with one row at a
          time and few names that repeat (no readout either); the preset then gets
          ``"feed": {"row_notices": true}``, so its rows that are no kills are read as
          notices;
    - ``popup``: such lines, but :data:`FEED_SCORE_ROWS` of them a "+" number
      alone (score popups: a value over a label, "+10" / "CRATE OPENED"); not read, and
      no extra notice comes from inside them;
    - ``banner``: large text (>= 3.4% of the frame height) near the centre that
      appears briefly;
    - ``notice``: other centred text lines (objective messages, hints);
    - ``dynamic``: anything else (reported, OCR'd with the generic vocabulary).
* Text that keeps appearing outside every covered region (the discovered ones and
  the big-text region, or an existing preset's regions) but too rarely for its
  cells to count as dynamic becomes up to 2 extra ``notice`` regions
  (:func:`notice_proposals`; ``extra_notice``, ``extra_notice2``, marked
  ``"origin": "outside"``): it must recur at a fixed screen anchor (left edge,
  right edge or centre) in 3+ bursts, and not as one or two texts that come back
  burst after burst and stay on screen (a label) or a few short readouts (a gauge). A
  new preset gets them with the rest; for a game with a preset they are only listed
  under ``suggested_regions`` in the proposal, to be copied into the preset by hand,
  and its own outside regions count against the 2.

The proposal is written as a preset JSON (see :mod:`cutpoint.events.regions`)
plus ``discovery/overlay.jpg`` (mean frame, dynamic-text heat map, proposed
rectangles) and the sampled frames as JPEGs, so the result can be checked by eye
and edited.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import time
from collections import Counter, deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.events.common import Source
from cutpoint.events.regions import Region, bigtext_region
from cutpoint.ffmpeg_utils import FFmpegError, grab_images, output_dir, probe

logger = logging.getLogger(__name__)

REF_W, REF_H = 1920, 1080
CELL = 24  # reference pixels per grid cell
GW, GH = REF_W // CELL, REF_H // CELL
MENU_CELL_W, MENU_CELL_H = REF_W // 8, REF_H // 6  # coarse 8 x 6 grid for the menu test
# Feed growth (_grow_feed): a feed's oldest rows show only when it is full, too rarely for
# their cells to count as dynamic. A row one pitch above or below the feed is added when
# text of the feed's height and x-span sits there in FEED_GROW_SHARE of the feed's frames
# (and FEED_GROW_BURSTS bursts) and varies (label variety >= FEED_GROW_VARIETY: no static
# label). A full feed is rare, so a row in fewer frames (FEED_GROW_MIN_FRAMES at least) is
# added too when its sightings differ (sighting variety >= FEED_GROW_VARIED: distinct
# normalised frame texts per burst; a static label seen in 2 bursts has 0.5) and at least
# FEED_GROW_FULL_SHARE of its frames show the full feed (an oldest row shows only then, a
# notice or chat line next to the feed any time). The feed grows to FEED_GROW_MAX times its
# height at most.
FEED_GROW_SHARE = 0.05
FEED_GROW_BURSTS = 2
FEED_GROW_VARIETY = 0.15  # the readout cut of cluster_regions
FEED_GROW_MIN_FRAMES = 2
FEED_GROW_VARIED = 0.75
FEED_GROW_FULL_SHARE = 0.5
FEED_GROW_MAX = 2.0
# A stack is no feed when FEED_SCORE_ROWS of its rows are a "+" number alone: a score
# popup stacks a value over a label ("+10" / "CRATE OPENED"), a feed's rows are names (a
# wrapped entry's tail such as "4-5)" or a crew count "2/3" has no sign). Measured: popups
# 0.42 and 0.47, feeds and every other stack 0.04 at most.
FEED_SCORE_ROWS = 0.3
_SCORE_RE = re.compile(r"\+\s?\d[\d,.]*")
# Icon kill lines ("name [weapon icon] name", no kill word): a line of two name-like boxes
# (_NAME_RE: one token, no ":" "%" "/" as in "FPS:106" or "3/4", half its letters and digits
# letters) split by a gap of FEED_ICON_GAP to FEED_ICON_GAP_MAX text heights. A side region
# of small text is a feed even with one row per frame and names that repeat (few players)
# when FEED_ICON_SHARE of its frames with text show such a line, in FEED_ICON_BURSTS bursts,
# with FEED_ICON_NAMES distinct names, FEED_ICON_BOTH of them on both sides of a line
# somewhere (players kill and die; a two-label HUD notice such as "Capture [icon] Alpha"
# keeps its verbs left and its targets right). Measured: R6 feeds 0.42 and 0.45 (6 and 10
# names, 2 of them on both sides), every other side region of R6, War Thunder, Destiny 2 and
# BF6 0.08 at most.
FEED_ICON_GAP = 1.0
FEED_ICON_GAP_MAX = 8.0
FEED_ICON_SHARE = 0.3
FEED_ICON_BURSTS = 3
FEED_ICON_NAMES = 4
FEED_ICON_BOTH = 1
_NAME_RE = re.compile(r"[^\s:%/]+")
# Outside notices (notice_proposals): text that keeps appearing outside every covered region.
# A left-over box is NOTICE_MIN_H px to NOTICE_MAX_H of the frame height tall; text in
# NOTICE_STATIC_SHARE of its source's frames, or on NOTICE_STATIC_CELLS static cells, is a
# label. Boxes join into a line across a gap of NOTICE_JOIN_GAP text heights. Lines of
# different frames cluster at heights within NOTICE_H_TOL and an edge or centre within
# NOTICE_ANCHOR_PX; a cluster needs NOTICE_MIN_FRAMES frames and fewer than
# NOTICE_SCOREBOARD_SHARE of them with a box-count jump. Rectangles stacked within
# NOTICE_MERGE_GAP text heights and overlapping by NOTICE_MERGE_OVERLAP merge; they are padded
# NOTICE_PAD_Y px and NOTICE_PAD_X of the width, and dropped over NOTICE_MAX_AREA of the frame
# or NOTICE_MAX_LINES lines. Text of NOTICE_BIG_TEXT_H px is detected at 0.6x. Two texts
# (lower-cased, digits left out) count as one when the shorter one, NOTICE_SAME_MIN characters
# or more, is part of the longer one or they match by NOTICE_SAME_RATIO (difflib ratio) with
# at most NOTICE_SAME_EDITS characters of the two unmatched: OCR misreads and cuts one text in
# many ways, while a fixed phrase with a different name is another text.
NOTICE_MIN_H = 10
NOTICE_MAX_H = 0.06
NOTICE_STATIC_SHARE = 0.45
NOTICE_STATIC_CELLS = 0.5
NOTICE_JOIN_GAP = 1.5
NOTICE_H_TOL = 0.25
NOTICE_ANCHOR_PX = 24
NOTICE_MIN_FRAMES = 3
NOTICE_SCOREBOARD_SHARE = 0.5
NOTICE_MERGE_GAP = 1.5
NOTICE_MERGE_OVERLAP = 0.5
NOTICE_PAD_Y = 12
NOTICE_PAD_X = 0.25
NOTICE_MAX_AREA = 0.06
NOTICE_MAX_LINES = 4
NOTICE_BIG_TEXT_H = 30
NOTICE_SAMPLES = 8
NOTICE_SAME_MIN = 4
NOTICE_SAME_RATIO = 0.8
NOTICE_SAME_EDITS = 3
NOTICE_BAND = 0.5 * NOTICE_MAX_H * REF_H  # cluster lookup band: half the tallest text


@dataclass
class DiscoverParams:
    """Sampling and clustering parameters."""

    bursts: int = 40
    burst_frames: int = 4
    burst_step_s: float = 1.5
    width: int = 1280  # decode width for OCR (text >= ~12 px at 1080p stays legible)
    min_score: float = 0.5
    # menus/scoreboards (see gameplay_frames): at least menu_min_boxes boxes and either
    # box centres in >= menu_cover of the centre cells of the coarse menu grid, or
    # >= menu_jump x the source's median boxes per frame
    menu_min_boxes: int = 24
    menu_cover: float = 0.5
    menu_jump: float = 2.0
    static_occupancy: float = 0.45  # cells with text this often...
    static_change: float = 0.12  # ...and changing less than this are static HUD
    min_change: float = 0.2  # dynamic cell: changes in >= 20% of its occupied pairs
    min_hits: int = 3  # dynamic cell: text seen in >= this many bursts...
    min_burst_share: float = 0.08  # ...and in >= 8% of the gameplay bursts
    min_occupancy: float = 0.03  # ...and in >= 3% of the gameplay frames
    # outside notices (see notice_proposals): a cluster of text outside every covered
    # region needs notice_min_bursts bursts; at most notice_max_regions are proposed,
    # counting the preset's own outside regions (0: off).
    # 3 bursts, not 2: world text (signs, name tags) lands within a text height of an
    # earlier sighting in some second burst by chance; in three, rarely
    notice_min_bursts: int = 3
    notice_max_regions: int = 2
    # a cluster whose lines are, for notice_steady_share of them, one or two texts (OCR
    # variants count as one) is a label when it shows in more than notice_steady_bursts
    # bursts per text (a map name, a lobby line, a gauge label; a notice with two messages
    # in 3 or 4 bursts stays) and in notice_steady_frames of the frames of those bursts (a
    # notice that recurs for a frame or so stays); stacked lines merged into one notice
    # are dropped when labels and gauges make up notice_steady_share of them
    notice_steady_share: float = 0.8
    notice_steady_bursts: float = 2.0
    notice_steady_frames: float = 0.3
    # ...and a gauge when notice_readout_share of its lines are readouts (at most
    # notice_readout_chars letters and digits, with a digit or mixed case: "RPM2100",
    # "ALTft"; plain words in any case, such as role names, are no readouts) of at most
    # notice_readout_texts texts with digits left out (a column of short names has more)
    notice_readout_chars: int = 8
    notice_readout_share: float = 0.5
    notice_readout_texts: int = 4


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
    bursts: list[tuple[Source, float]],
    params: DiscoverParams,
    frame_dir: Path,
    cache_path: Path,
    on_frame: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """OCR every burst frame (cached in ``cache_path`` as JSON lines); ``on_frame`` is
    called with (frames done, frames planned) after each frame and each burst."""
    import cv2  # noqa: PLC0415

    from cutpoint.events.killfeed import OcrEngine  # noqa: PLC0415

    done = {rec["id"]: rec for rec in read_ocr_samples(cache_path)}
    total = len(bursts) * params.burst_frames
    report = on_frame or (lambda _done, _total: None)
    engine = None
    out: list[dict[str, Any]] = []
    fh = None
    # a frame's id is its source and time; bursts overlap on a short source, so a time
    # taken by an earlier frame gets the burst and frame number too (unique and stable)
    seen: set[str] = set()
    plan_ids = []
    for bi, (src, t) in enumerate(bursts):
        ids = []
        for k in range(params.burst_frames):
            fid = f"{src.name}@{t + k * params.burst_step_s:.1f}"
            fid = f"{fid}#{bi}.{k}" if fid in seen else fid
            seen.add(fid)
            ids.append(fid)
        plan_ids.append(ids)

    def cached(fid: str, t: float, bi: int, k: int) -> dict[str, Any] | None:
        """The cached record of this frame: same id and time (an older cache can hold
        another burst's frame under a shared id), in this run's burst."""
        rec = done.get(fid)
        if rec is None or rec.get("t") != round(t + k * params.burst_step_s, 2):
            return None
        return rec | {"burst": bi, "k": k}

    # failed before the first frame is saved: no empty frames/ (or empty ocr.jsonl) left
    # behind; the cache is opened on the first record and closed before output_dir cleans up
    with output_dir(frame_dir), ExitStack() as stack:
        for bi, (src, t) in enumerate(bursts):
            ids = plan_ids[bi]
            hits = [cached(fid, t, bi, k) for k, fid in enumerate(ids)]
            if all(h is not None for h in hits):
                out += [h for h in hits if h is not None]
                report((bi + 1) * params.burst_frames, total)
                continue
            frames = _burst_frames(
                src.io_path, t, params.burst_frames, params.burst_step_s, params.width
            )
            if engine is None:
                # full frames, one at a time: all the threads on one engine
                engine = OcrEngine(det_side=params.width, lanes=1)
            for k, img in enumerate(frames):
                fid, hit = ids[k], hits[k]
                if hit is not None:
                    out.append(hit)
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
                report(bi * params.burst_frames + k + 1, total)
            report((bi + 1) * params.burst_frames, total)  # a short burst counts as done
            if (bi + 1) % 10 == 0:
                logger.info("  discovery OCR: %d/%d bursts", bi + 1, len(bursts))
    return out


# --------------------------------------------------------------------------- menus


def _centre_cover(frame: dict[str, Any]) -> float:
    """Share of the 24 centre cells (6 x 4 inside the coarse 8 x 6 grid) that hold
    a text box centre: HUDs hug the edges and keep the centre free for the game
    view; menus and scoreboards fill it."""
    cells = set()
    for b in frame.get("boxes", []):
        gx = int((b["x0"] + b["x1"]) / 2 // MENU_CELL_W)
        gy = int((b["y0"] + b["y1"]) / 2 // MENU_CELL_H)
        if 1 <= gx <= 6 and 1 <= gy <= 4:
            cells.add((gx, gy))
    return len(cells) / 24


def gameplay_frames(frames: list[dict[str, Any]], params: DiscoverParams) -> list[dict[str, Any]]:
    """``frames`` without menu and scoreboard frames, in order.

    A frame is a menu when it has at least ``params.menu_min_boxes`` text boxes
    and either their centres fall in at least ``params.menu_cover`` of the
    centre cells (:func:`_centre_cover`) or it has at least ``params.menu_jump``
    times the median box count of its source's frames. A box count alone says
    nothing: a busy HUD shows 50 to 100 boxes in every frame. Discovery and the
    game profile builder use this test, so both see the same frames.
    """
    counts: dict[str, list[int]] = {}
    for f in frames:
        counts.setdefault(str(f.get("source")), []).append(len(f.get("boxes", [])))
    median = {s: float(np.median(c)) for s, c in counts.items()}
    out = []
    for f in frames:
        n = len(f.get("boxes", []))
        menu = n >= params.menu_min_boxes and (
            _centre_cover(f) >= params.menu_cover
            or n >= params.menu_jump * median[str(f.get("source"))]
        )
        if not menu:
            out.append(f)
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


def _lines(bs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The boxes of one frame grouped into lines, each in x order: boxes whose y-ranges
    overlap by half the shorter box's height or more share a line."""
    lines: list[list[dict[str, Any]]] = []
    for b in sorted(bs, key=lambda b: b["y0"]):
        h = b["y1"] - b["y0"]
        for ln in lines:
            if any(
                min(b["y1"], o["y1"]) - max(b["y0"], o["y0"]) >= 0.5 * min(h, o["y1"] - o["y0"])
                for o in ln
            ):
                ln.append(b)
                break
        else:
            lines.append([b])
    return [sorted(ln, key=lambda o: o["x0"]) for ln in lines]


def _score_rows(frames: list[list[dict[str, Any]]]) -> float:
    """Share of the lines whose text is a "+" number alone ("+10", "+1,000"). A line is
    the boxes of one frame whose y-ranges overlap by half the shorter box's height or
    more (:func:`_lines`), joined in x order, so a shorter or bottom-aligned "+N" box next
    to a name stays on the name's line instead of counting as a row of its own."""
    rows = ["".join(o["text"] for o in ln) for bs in frames for ln in _lines(bs)]
    return sum(bool(_SCORE_RE.fullmatch(t.strip())) for t in rows) / max(1, len(rows))


def _name_like(text: str) -> bool:
    """One token of 3+ letters and digits, half of them letters, without ":" "%" "/"."""
    t = text.strip()
    n = sum(c.isalnum() for c in t)
    return n >= 3 and sum(c.isalpha() for c in t) >= 0.5 * n and bool(_NAME_RE.fullmatch(t))


def _icon_lines(
    inside: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[float, int, int, int]:
    """Icon kill lines in a region: (share of its frames with one, bursts, distinct names,
    names seen both left and right of an icon gap).

    An icon line ("name [weapon icon] name", no kill word) has two adjacent name-like
    boxes (:func:`_name_like`) split by a gap of :data:`FEED_ICON_GAP` to
    :data:`FEED_ICON_GAP_MAX` times their median height. Names count lower-cased, letters
    only."""
    hits, bursts, left, right = 0, set(), set(), set()
    for f, bs in inside:
        found = False
        for ln in _lines(bs):
            for a, b in pairwise(ln):
                h = float(np.median([a["y1"] - a["y0"], b["y1"] - b["y0"]]))
                gap = b["x0"] - a["x1"]
                if (
                    FEED_ICON_GAP * h <= gap <= FEED_ICON_GAP_MAX * h
                    and _name_like(a["text"])
                    and _name_like(b["text"])
                ):
                    found = True
                    left.add(re.sub(r"[^a-z]", "", a["text"].lower()))
                    right.add(re.sub(r"[^a-z]", "", b["text"].lower()))
        if found:
            hits += 1
            bursts.add((f["source"], f["burst"]))
    names, both = (left | right) - {""}, (left & right) - {""}
    return hits / max(1, len(inside)), len(bursts), len(names), len(both)


def _sighting_variety(band: list[tuple[str, tuple[str, int], float, str]]) -> float:
    """Distinct texts of a row's sightings per burst, at most 1. A sighting is the
    texts of one frame in ``band`` (``(frame, burst, centre y, text)``), each
    lower-cased, without digits and punctuation, joined in sorted order; so a row that
    stays for a few frames or is split into boxes (in any order) counts once per text,
    and a static label (in any case) has 1 / bursts."""
    texts: dict[str, list[str]] = {}
    for fid, _, _, t in band:
        texts.setdefault(fid, []).append(re.sub(r"[0-9]", "", _norm_text(t)))
    distinct = {"".join(sorted(ts)) for ts in texts.values()} - {""}
    return min(1.0, len(distinct) / max(1, len({bk for _, bk, _, _ in band})))


def _grow_feed(
    frames: list[dict[str, Any]], rect: tuple[int, int, int, int], med_h: float, n_feed: int
) -> tuple[int, int, int]:
    """Grow a feed rectangle ``(x0, y0, x1, y1)`` up and down by whole rows.

    Rows are text boxes centred in the feed's x-span, 0.5 to 2 times its median
    text height ``med_h``. The row pitch is the median gap between adjacent
    rows of a frame inside the feed. The feed's top (bottom) row is the outermost
    row inside it with the support a grown row needs (:func:`_outer_row`), so a
    stray box near the edge does not set it. One pitch past the top (bottom) row
    a row is added when its boxes (centre within a third of a pitch) show in at
    least :data:`FEED_GROW_SHARE` of the ``n_feed`` frames with feed text and in
    :data:`FEED_GROW_BURSTS` bursts and their texts vary
    (:data:`FEED_GROW_VARIETY`). A row in fewer frames, but at least
    :data:`FEED_GROW_MIN_FRAMES`, is added when its sightings differ
    (:func:`_sighting_variety` at least :data:`FEED_GROW_VARIED`) and at least
    :data:`FEED_GROW_FULL_SHARE` of its frames show the full feed (the feed's row next
    to it and a row at every other pitch from its top to its bottom row but one, as OCR
    misses a box now and then). Growth stops at :data:`FEED_GROW_MAX` times
    the height. One DEBUG line says which condition added each row ("ok: frames" or
    "ok: varied") and why growth stopped in each direction.

    Returns:
        (y0, y1, rows added).
    """
    x0, y0, x1, y1 = rect
    rows: list[tuple[str, tuple[str, int], float, str]] = []  # (frame, burst, centre y, text)
    gaps: list[float] = []
    for f in frames:
        cys = []
        for b in f["boxes"]:
            cy, h = (b["y0"] + b["y1"]) / 2, b["y1"] - b["y0"]
            if x0 <= (b["x0"] + b["x1"]) / 2 <= x1 and 0.5 * med_h <= h <= 2 * med_h:
                rows.append((str(f["id"]), (f["source"], f["burst"]), cy, b["text"]))
                if y0 <= cy <= y1:
                    cys.append(cy)
        cys.sort()
        gaps += [b - a for a, b in pairwise(cys) if 0.5 * med_h < b - a < 3 * med_h]
    where = f"feed {x0},{y0} {x1 - x0}x{y1 - y0}"
    if not gaps:
        logger.debug("Feed growth %s: no two rows in one frame, no pitch; 0 rows added", where)
        return y0, y1, 0
    pitch = float(np.median(gaps))
    need = max(2.0, FEED_GROW_SHARE * n_feed)
    inner = sorted((cy, fid) for fid, _, cy, _ in rows if y0 <= cy <= y1)
    top, bottom = _outer_row(inner, pitch, need, -1), _outer_row(inner, pitch, need, 1)
    limit = FEED_GROW_MAX * (y1 - y0)
    ends = {-1: (top, y0), 1: (bottom, y1)}  # direction -> (outer row centre, edge)
    # a frame shows the full feed when it has a row at every pitch from top to bottom
    slots = round((bottom - top) / pitch) + 1
    per_frame: dict[str, set[int]] = {}
    for cy, fid in inner:
        per_frame.setdefault(fid, set()).add(round((cy - top) / pitch))
    full_ids = {fid for fid, s in per_frame.items() if len(s & set(range(slots))) >= slots}
    # for growth past the top (bottom) row a frame counts as full when it has that row and
    # misses at most one other: OCR misses a box now and then
    near_full = {
        d: {
            fid
            for fid, s in per_frame.items()
            if end in s and len(s & set(range(slots))) >= slots - 1
        }
        for d, end in ((-1, 0), (1, slots - 1))
    }
    added = 0
    why: dict[int, list[str]] = {-1: [], 1: []}
    for d in (-1, 1):
        while True:
            centre = ends[d][0] + d * pitch
            band = [row for row in rows if abs(row[2] - centre) <= pitch / 3]
            edge = centre + d * pitch / 2
            n_frames = len({fid for fid, _, _, _ in band})
            n_bursts = len({bk for _, bk, _, _ in band})
            n_full = len({fid for fid, _, _, _ in band} & near_full[d])
            variety = _label_variety([t for _, _, _, t in band])
            seen = _sighting_variety(band)
            # a full feed is rare: fewer frames do when the row's sightings differ and
            # show with the full feed
            varied = (
                n_frames >= FEED_GROW_MIN_FRAMES
                and seen >= FEED_GROW_VARIED
                and n_full >= FEED_GROW_FULL_SHARE * n_frames
            )
            checks = (
                ("limit", abs(edge - ends[-d][1]) > limit),
                ("screen edge", not 0 <= edge <= REF_H),
                ("frames", n_frames < need and not varied),
                ("bursts", n_bursts < FEED_GROW_BURSTS),
                ("variety", variety < FEED_GROW_VARIETY),
            )
            stop = next((name for name, hit in checks if hit), None)
            row = (
                f"row at y {centre:.0f}: {n_frames} frames ({n_full} full), {n_bursts} bursts, "
                f"variety {variety:.2f}, sighting variety {seen:.2f}, "
            )
            if stop is not None:
                why[d].append(f"{row}stop: {stop}")
                break
            why[d].append(f"{row}ok: {'frames' if n_frames >= need else 'varied'}")
            centre = float(np.median([cy for _, _, cy, _ in band]))
            ends[d] = (centre, round(centre + d * pitch / 2))
            added += 1
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Feed growth %s: pitch %.1f px, %d rows, full in %d/%d frames; a row needs "
            "%.0f frames (%d if sighting variety >= %.2f and %.0f%% full), %d bursts, "
            "variety %.2f; above: %s; below: %s; %d rows added",
            where,
            pitch,
            slots,
            len(full_ids),
            n_feed,
            need,
            FEED_GROW_MIN_FRAMES,
            FEED_GROW_VARIED,
            100 * FEED_GROW_FULL_SHARE,
            FEED_GROW_BURSTS,
            FEED_GROW_VARIETY,
            " | ".join(why[-1]),
            " | ".join(why[1]),
            added,
        )
    return min(y0, int(ends[-1][1])), max(y1, int(ends[1][1])), added


def _outer_row(inner: list[tuple[float, str]], pitch: float, need: float, d: int) -> float:
    """Centre of the top (``d`` = -1) or bottom (1) row of ``inner`` (``(centre y,
    frame)`` sorted by y): the first row from that end whose boxes within a third of a
    pitch show in ``need`` frames, re-centred on their median. Falls back to the
    outermost box when no row has the support."""
    cys = [cy for cy, _ in inner]
    order = range(len(cys)) if d == -1 else range(len(cys) - 1, -1, -1)
    for i in order:
        lo = bisect.bisect_left(cys, cys[i] - pitch / 3)
        hi = bisect.bisect_right(cys, cys[i] + pitch / 3)
        if len({fid for _, fid in inner[lo:hi]}) >= need:
            c = float(np.median(cys[lo:hi]))
            lo = bisect.bisect_left(cys, c - pitch / 3)
            hi = bisect.bisect_right(cys, c + pitch / 3)
            return float(np.median(cys[lo:hi]))
    return cys[0] if d == -1 else cys[-1]


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
    game_frames = gameplay_frames(frames, params)
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
        scores = _score_rows([bs for _, bs in inside])
        cx = (x0 + x1) / 2 / REF_W
        med_h = float(np.median(heights))
        mean_rows = float(np.mean(rows))
        appear = len(inside) / n
        variety = _label_variety(texts)
        alpha = _alpha_share(texts)
        side = min(cx, 1 - cx) < 0.3  # feeds hug the left/right edge
        icon_share, icon_bursts, icon_names, icon_both = _icon_lines(inside)
        # a feed of icon kill lines shows one or two rows, and its few names repeat
        icon_feed = (
            side
            and med_h < 0.034 * REF_H
            and icon_share >= FEED_ICON_SHARE
            and icon_bursts >= FEED_ICON_BURSTS
            and icon_names >= FEED_ICON_NAMES
            and icon_both >= FEED_ICON_BOTH
        )
        if (variety < 0.15 or alpha < 0.45) and not icon_feed:
            kind = "readout"  # gauges, compass, ammo: labels fixed, numbers change
        elif (mean_rows >= 1.3 or icon_feed) and side and med_h < 0.034 * REF_H:
            if scores >= FEED_SCORE_ROWS:
                kind = "popup"  # score popups: a value over a label, not read
            else:
                kind = "feed"
                y0, y1, grown = _grow_feed(game_frames, (x0, y0, x1, y1), med_h, len(inside))
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
                    "score_rows": round(scores, 3),
                    "icon_lines": round(icon_share, 3),
                    "icon_names": [icon_names, icon_both],
                    "icon_feed": icon_feed,
                    "label_variety": round(variety, 3),
                    "alpha_share": round(alpha, 3),
                    "change_rate": round(float(chg_rate[gy0:gy1, gx0:gx1].mean()), 3),
                    "samples": texts[:8],
                }
                | ({"grown_rows": grown} if kind == "feed" else {}),
            }
        )
    if not any(p["kind"] == "banner" for p in proposals):
        ban = _banner_proposal(game_frames, 0.034 * REF_H)
        if ban:
            proposals.append(ban)
    order = {"feed": 0, "banner": 1, "notice": 2, "dynamic": 3, "readout": 4, "popup": 5}
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


# --------------------------------------------------------------------------- outside notices


def _height_gated(r: Region) -> bool:
    """Regions the OCR layer reads only text at least ``min_text_h`` tall in: any
    det-gated region with ``min_text_h`` (its text detection drops shorter boxes) and
    big-text and banner regions with it (their interpretation skips shorter lines),
    such as the generic big-text check. Smaller text inside them is not read."""
    return r.min_text_h > 0 and (r.gate == "det" or r.kind in ("bigtext", "banner"))


def _is_covered(b: dict[str, Any], covered: tuple[Region, ...]) -> bool:
    """Whether a region the OCR layer reads covers box ``b``: its centre lies in the
    region and, for a height-gated one (:func:`_height_gated`), it is tall enough."""
    cx, cy, h = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, b["y1"] - b["y0"]
    return any(
        r.x <= cx <= r.x + r.w
        and r.y <= cy <= r.y + r.h
        and (not _height_gated(r) or h >= r.min_text_h)
        for r in covered
    )


def _static_key(text: str) -> str:
    return re.sub(r"[0-9]", "", _norm_text(text))


def _notice_letters(text: str) -> bool:
    """The OCR layer's rule for an unknown notice line: 4+ letters, 60% of the
    characters (spaces left out)."""
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 4 and letters >= 0.6 * len(text.replace(" ", ""))


def _same_text(a: str, b: str) -> bool:
    """Whether two :func:`_static_key` texts are one text read differently: the shorter
    one (:data:`NOTICE_SAME_MIN` characters or more) is part of the longer one, or they
    match by :data:`NOTICE_SAME_RATIO` with at most :data:`NOTICE_SAME_EDITS` characters
    of the two left unmatched (so "<phrase> <name>" with another name is another text)."""
    short, long = sorted((a, b), key=len)
    if len(short) >= NOTICE_SAME_MIN and short in long:
        return True
    if len(long) - len(short) > NOTICE_SAME_EDITS:  # at least this many left unmatched
        return False
    sm = SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(m.size for m in sm.get_matching_blocks())
    return sm.ratio() >= NOTICE_SAME_RATIO and len(a) + len(b) - 2 * matched <= NOTICE_SAME_EDITS


def _text_groups(keys: list[str]) -> list[int]:
    """Line counts of the texts among ``keys`` (:func:`_static_key` per line, empty ones
    left out), with texts that :func:`_same_text` matches counted as one, largest first.
    Texts join the first group whose most frequent text they match, most frequent first."""
    reps: list[str] = []
    sizes: list[int] = []
    for key, n in sorted(Counter(k for k in keys if k).items(), key=lambda kv: (-kv[1], kv[0])):
        i = next((i for i, r in enumerate(reps) if _same_text(key, r)), None)
        if i is None:
            reps.append(key)
            sizes.append(n)
        else:
            sizes[i] += n
    return sorted(sizes, reverse=True)


def _steady(lines: list[dict[str, Any]], params: DiscoverParams) -> bool:
    """Whether a cluster's ``lines`` are a label, not a notice: one or two texts
    (:func:`_text_groups`) make up ``params.notice_steady_share`` of them, the cluster
    shows in more than ``params.notice_steady_bursts`` bursts per text and it stays: in
    at least ``params.notice_steady_frames`` of the frames of its bursts (a notice that
    recurs shows for a frame or so per burst)."""
    sizes = _text_groups([_static_key(ln["text"]) for ln in lines])
    total = sum(sizes)
    if not total:
        return False
    texts = next((k for k in (1, 2) if sum(sizes[:k]) >= params.notice_steady_share * total), None)
    bursts = len({ln["bkey"] for ln in lines})
    frames = len({ln["fid"] for ln in lines})
    return (
        texts is not None
        and bursts > params.notice_steady_bursts * texts
        and frames >= params.notice_steady_frames * params.burst_frames * bursts
    )


def _readout(text: str, max_chars: int) -> bool:
    """Whether a line is a gauge readout: at most ``max_chars`` letters and digits, with
    a digit or a word in mixed case (both cases, not capitalised; scripts without case
    have no mixed-case words)."""
    chars = [ch for ch in text if ch.isalnum()]
    if not chars or len(chars) > max_chars:
        return False
    if any(ch.isdigit() for ch in chars):
        return True
    words = re.findall(r"[^\W\d_]+", text)
    return any(
        any(c.isupper() for c in w) and any(c.islower() for c in w) and not w.istitle()
        for w in words
    )


def _frame_lines(boxes: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Boxes of one frame joined into text lines: centres within half a text height
    and a gap of at most 1.5 text heights."""
    lines: list[dict[str, Any]] = []
    for b in sorted(boxes, key=lambda b: (b["x0"], b["y0"], b["text"])):
        h, cy = b["y1"] - b["y0"], (b["y0"] + b["y1"]) / 2
        for ln in lines:
            near = b["x0"] - ln["x1"] <= NOTICE_JOIN_GAP * max(h, ln["h"])
            if near and abs(cy - ln["cy"]) <= 0.5 * min(h, ln["h"]):
                ln["x1"] = max(ln["x1"], b["x1"])
                ln["y0"], ln["y1"] = min(ln["y0"], b["y0"]), max(ln["y1"], b["y1"])
                ln["hs"].append(h)
                ln["texts"].append(b["text"])
                break
        else:
            lines.append(
                {"x0": b["x0"], "x1": b["x1"], "y0": b["y0"], "y1": b["y1"], "cy": cy, "h": h}
                | {"hs": [h], "texts": [b["text"]]}
            )
    return [
        {
            "x0": ln["x0"],
            "x1": ln["x1"],
            "y0": ln["y0"],
            "y1": ln["y1"],
            "cy": (ln["y0"] + ln["y1"]) / 2,
            "cx": (ln["x0"] + ln["x1"]) / 2,
            "h": float(np.median(ln["hs"])),
            "text": " ".join(ln["texts"]).strip(),
        }
        for ln in lines
    ]


def _cluster_stats(lines: list[dict[str, Any]], n_frames: int) -> dict[str, Any]:
    """Evidence of a group of lines (``fid``, ``bkey`` and ``text`` per line)."""
    frames = {ln["fid"] for ln in lines}
    texts = {_static_key(ln["text"]) for ln in lines} - {""}
    spread = {
        "left": float(np.std([ln["x0"] for ln in lines])),
        "centre": float(np.std([ln["cx"] for ln in lines])),
        "right": float(np.std([ln["x1"] for ln in lines])),
    }
    # the steadiest edge is the anchor; a tie (one text) counts as centred
    anchor = min(spread, key=lambda a: (round(spread[a], 1), ("centre", "left", "right").index(a)))
    return {
        "bursts": len({ln["bkey"] for ln in lines}),
        "sources": len({ln["bkey"][0] for ln in lines}),
        "frames": len(frames),
        "distinct": len(texts),
        "share": round(len(frames) / max(1, n_frames), 3),
        "anchor": anchor,
        "median_text_h": round(float(np.median([ln["h"] for ln in lines])), 1),
    }


def _core_rect(lines: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """The 5th to 95th percentile of the line edges: ``(x0, y0, x1, y1)``."""
    return _edges_rect(_edges(lines))


def _edges(lines: list[dict[str, Any]]) -> np.ndarray:
    """Line edges and heights as rows ``(x0, y0, x1, y1, h)``."""
    return np.array([(ln["x0"], ln["y0"], ln["x1"], ln["y1"], ln["h"]) for ln in lines], float)


def _edges_rect(edges: np.ndarray) -> tuple[int, int, int, int]:
    """:func:`_core_rect` of :func:`_edges` rows."""
    x0, y0 = np.floor(np.percentile(edges[:, :2], 5, axis=0))
    x1, y1 = np.ceil(np.percentile(edges[:, 2:4], 95, axis=0))
    return int(x0), int(y0), int(x1), int(y1)


def _trim(
    rect: tuple[int, int, int, int], r: Region, keep: tuple[float, float]
) -> tuple[int, int, int, int] | None:
    """``rect`` cut so it no longer overlaps region ``r``: the largest of the four cuts
    that still holds the point ``keep``; None when none does."""
    x0, y0, x1, y1 = rect
    rx0, ry0, rx1, ry1 = r.x, r.y, r.x + r.w, r.y + r.h
    if rx0 >= x1 or rx1 <= x0 or ry0 >= y1 or ry1 <= y0:
        return rect
    cuts = [(x0, y0, x1, ry0), (x0, ry1, x1, y1), (x0, y0, rx0, y1), (rx1, y0, x1, y1)]
    ok = [
        c
        for c in cuts
        if c[0] < c[2] and c[1] < c[3] and c[0] <= keep[0] <= c[2] and c[1] <= keep[1] <= c[3]
    ]
    return max(ok, key=lambda c: (c[2] - c[0]) * (c[3] - c[1])) if ok else None


def _merge_stacked(clusters: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """``clusters`` with the stacked lines of one notice merged: two clusters merge when
    their rectangles (:func:`_core_rect`) are within :data:`NOTICE_MERGE_GAP` times the
    larger median text height of each other vertically and overlap by
    :data:`NOTICE_MERGE_OVERLAP` of the narrower width horizontally.

    Each cluster's rectangle and height are computed once; after a merge (into the
    earlier cluster, lines in order) only the merged cluster is computed and checked
    again, so the work grows with the square of the clusters, not the cube. In order.
    """
    members = dict(enumerate(clusters))
    edges = {i: _edges(c) for i, c in members.items()}
    rect = {i: _edges_rect(e) for i, e in edges.items()}
    med = {i: float(np.median(e[:, 4])) for i, e in edges.items()}

    def stacked(i: int, j: int) -> bool:
        a, b = rect[i], rect[j]
        gap = max(a[1], b[1]) - min(a[3], b[3])
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        return gap <= NOTICE_MERGE_GAP * max(med[i], med[j]) and (
            overlap >= NOTICE_MERGE_OVERLAP * min(a[2] - a[0], b[2] - b[0])
        )

    todo = deque(members)
    while todo:
        i = todo.popleft()
        if i not in members:
            continue  # merged into another one
        j = next((j for j in members if j != i and stacked(i, j)), None)
        if j is None:
            continue
        lo, hi = min(i, j), max(i, j)
        members[lo] = members[lo] + members.pop(hi)
        edges[lo] = np.concatenate((edges[lo], edges.pop(hi)))
        rect[lo] = _edges_rect(edges[lo])
        med[lo] = float(np.median(edges[lo][:, 4]))
        todo.appendleft(lo)  # only the merged cluster changed: check it again
    return list(members.values())


def notice_proposals(
    game_frames: list[dict[str, Any]],
    covered: tuple[Region, ...],
    params: DiscoverParams,
    static_mask: np.ndarray,
    existing: int = 0,
) -> list[dict[str, Any]]:
    """Extra ``notice`` regions where text keeps appearing outside every covered region.

    Works on the OCR'd gameplay frames discovery already has (no OCR of its own):

    * A box is left over when its centre lies outside every region in ``covered``
      (inside a height-gated one, :func:`_height_gated`: a det-gated region with
      ``min_text_h`` or a big-text or banner region with it, such as the big-text
      check, also when it is shorter than the region's ``min_text_h``: that text is
      not read there), it is
      :data:`NOTICE_MIN_H` px to :data:`NOTICE_MAX_H` of the frame height tall, its
      text (lower-cased, digits left out) is in fewer than
      :data:`NOTICE_STATIC_SHARE` of its source's gameplay frames and fewer than
      :data:`NOTICE_STATIC_CELLS` of its cells are static in ``static_mask``.
    * Left-over boxes on one row of a frame are joined into lines; a line needs the
      OCR layer's notice letter rule (4+ letters, 60% letters).
    * Lines of different frames form a cluster when their centres are within half a
      text height, their heights within :data:`NOTICE_H_TOL` of each other and the
      left edge, right edge or centre within :data:`NOTICE_ANCHOR_PX` px: a notice
      stays at a fixed screen anchor while its text changes. A line joins the
      matching cluster with the nearest centre (the first on a tie); clusters are
      looked up in bands of :data:`NOTICE_BAND` px, the most half a text height can
      reach.
    * A cluster needs ``params.notice_min_bursts`` bursts and :data:`NOTICE_MIN_FRAMES`
      frames, and fewer than :data:`NOTICE_SCOREBOARD_SHARE` of its frames with
      ``params.menu_jump`` times the source's median box count (a partial scoreboard).
    * A notice changes text. A cluster is a label (:func:`_steady`) when one or two
      texts, OCR variants counted as one (:func:`_text_groups`), make up
      ``params.notice_steady_share`` of its lines and it shows in more than
      ``params.notice_steady_bursts`` bursts per text, in at least
      ``params.notice_steady_frames`` of the frames of those bursts; it is a gauge when
      ``params.notice_readout_share`` of its lines are readouts (:func:`_readout`:
      ``params.notice_readout_chars`` letters and digits at most, with a digit or mixed
      case) of at most ``params.notice_readout_texts`` texts (digits left out).
    * Its rectangle spans the 5th to 95th percentile of the line edges; rectangles
      stacked within :data:`NOTICE_MERGE_GAP` text heights that overlap horizontally
      by :data:`NOTICE_MERGE_OVERLAP` merge, labels and gauges too; a merged rectangle
      is dropped when labels and gauges make up ``params.notice_steady_share`` of its
      lines (a block of menu labels). Padding: :data:`NOTICE_PAD_Y` px up and
      down, :data:`NOTICE_PAD_X` of the width on the side(s) the text grows to (both
      for centred text, right for left-aligned, left for right-aligned). A rectangle
      over :data:`NOTICE_MAX_AREA` of the frame or taller than
      :data:`NOTICE_MAX_LINES` lines is dropped; overlaps with covered regions that
      are not height-gated are cut off.
    * The ``params.notice_max_regions`` best (bursts x distinct texts), less the
      ``existing`` outside regions the covered preset already has, are named
      ``extra_notice``, ``extra_notice2``, ... (never a name in ``covered``).

    Returns:
        Proposals ``{name, kind: "notice", x, y, w, h, score, evidence}`` (the
        final, padded rectangle; ``evidence["outside"]`` is True), best first.
    """
    if params.notice_max_regions <= existing or not game_frames:
        return []
    n_frames = len(game_frames)
    per_source: dict[str, list[dict[str, Any]]] = {}
    for f in game_frames:
        per_source.setdefault(str(f["source"]), []).append(f)
    median = {s: float(np.median([len(f["boxes"]) for f in fs])) for s, fs in per_source.items()}
    static_texts: dict[str, set[str]] = {}
    for s, fs in per_source.items():
        seen: dict[str, int] = {}
        for f in fs:
            for key in {_static_key(b["text"]) for b in f["boxes"]} - {""}:
                seen[key] = seen.get(key, 0) + 1
        static_texts[s] = {k for k, c in seen.items() if c >= NOTICE_STATIC_SHARE * len(fs)}
    busy: set[str] = set()  # frames with a box-count jump (partial scoreboards)
    lines: list[dict[str, Any]] = []
    order = sorted(
        game_frames, key=lambda f: (str(f["source"]), int(f["burst"]), int(f.get("k", 0)))
    )
    for f in order:
        src = str(f["source"])
        if median[src] > 0 and len(f["boxes"]) >= params.menu_jump * median[src]:
            busy.add(str(f["id"]))
        left = []
        for b in f["boxes"]:
            h = b["y1"] - b["y0"]
            if not NOTICE_MIN_H <= h <= NOTICE_MAX_H * REF_H or _is_covered(b, covered):
                continue
            if _static_key(b["text"]) in static_texts[src]:
                continue
            if static_mask[_cells(b)].mean() >= NOTICE_STATIC_CELLS:
                continue
            left.append(b)
        for ln in _frame_lines(left):
            if _notice_letters(str(ln["text"])):
                lines.append(ln | {"fid": str(f["id"]), "bkey": (src, int(f["burst"]))})
    clusters: list[list[dict[str, Any]]] = []
    means: list[dict[str, float]] = []
    # cluster indices by band of their mean centre: a match is within half a text height
    # (at most NOTICE_BAND px), so only the line's band and its two neighbours can hold one
    bands: dict[int, set[int]] = {}
    for ln in lines:
        best = None
        band = int(ln["cy"] // NOTICE_BAND)
        near = sorted(i for d in (-1, 0, 1) for i in bands.get(band + d, ()))
        for i in near:
            m = means[i]
            if (
                abs(ln["cy"] - m["cy"]) <= 0.5 * m["h"]
                and abs(ln["h"] - m["h"]) <= NOTICE_H_TOL * max(ln["h"], m["h"])
                and min(abs(ln[a] - m[a]) for a in ("x0", "x1", "cx")) <= NOTICE_ANCHOR_PX
            ) and (best is None or abs(ln["cy"] - m["cy"]) < abs(ln["cy"] - means[best]["cy"])):
                best = i
        if best is None:
            best = len(clusters)
            clusters.append([ln])
            means.append({a: float(ln[a]) for a in ("cy", "h", "x0", "x1", "cx")})
        else:
            bands[int(means[best]["cy"] // NOTICE_BAND)].discard(best)
            clusters[best].append(ln)
            n = len(clusters[best])
            for a in means[best]:
                means[best][a] += (ln[a] - means[best][a]) / n
        bands.setdefault(int(means[best]["cy"] // NOTICE_BAND), set()).add(best)
    candidates: list[list[dict[str, Any]]] = []
    why: dict[str, int] = {}
    for c in clusters:
        frames = {ln["fid"] for ln in c}
        checks = (
            ("bursts", len({ln["bkey"] for ln in c}) < params.notice_min_bursts),
            ("frames", len(frames) < NOTICE_MIN_FRAMES),
            ("scoreboard", len(frames & busy) >= NOTICE_SCOREBOARD_SHARE * len(frames)),
        )
        stop = next((name for name, hit in checks if hit), None)
        if stop is not None:
            why[stop] = why.get(stop, 0) + 1
            continue
        # a label or gauge line still merges with a stacked notice (a fixed heading over a
        # changing line) but is dropped on its own or when labels make up the merged one
        readouts = [
            _static_key(str(ln["text"]))
            for ln in c
            if _readout(str(ln["text"]), params.notice_readout_chars)
        ]
        gauge = (
            len(readouts) >= params.notice_readout_share * len(c)
            and len(_text_groups(readouts)) <= params.notice_readout_texts
        )
        fail = "steady" if _steady(c, params) else "readout" if gauge else None
        candidates.append([ln | {"fail": fail} for ln in c])
    kept: list[list[dict[str, Any]]] = []
    for c in _merge_stacked(candidates):
        fails = [ln["fail"] for ln in c if ln["fail"]]
        if len(fails) == len(c):
            why[fails[0]] = why.get(fails[0], 0) + 1
        elif len(fails) >= params.notice_steady_share * len(c):  # a block of labels
            why["labels"] = why.get("labels", 0) + 1
        else:
            kept.append(c)
    trims = tuple(r for r in covered if not _height_gated(r))
    out: list[dict[str, Any]] = []
    for c in kept:
        st = _cluster_stats(c, n_frames)
        x0, y0, x1, y1 = _core_rect(c)
        med_h = st["median_text_h"]
        if y1 - y0 > NOTICE_MAX_LINES * NOTICE_MERGE_GAP * med_h:
            why["tall"] = why.get("tall", 0) + 1
            continue
        keep = ((x0 + x1) / 2, (y0 + y1) / 2)
        pad = NOTICE_PAD_X * (x1 - x0)
        grow_left = pad if st["anchor"] in ("centre", "right") else 0
        grow_right = pad if st["anchor"] in ("centre", "left") else 0
        rect: tuple[int, int, int, int] | None = (
            max(0, round(x0 - grow_left)),
            max(0, y0 - NOTICE_PAD_Y),
            min(REF_W, round(x1 + grow_right)),
            min(REF_H, y1 + NOTICE_PAD_Y),
        )
        for r in trims:
            if rect is not None:
                rect = _trim(rect, r, keep)
        if rect is None:
            why["covered"] = why.get("covered", 0) + 1
            continue
        if (rect[2] - rect[0]) * (rect[3] - rect[1]) > NOTICE_MAX_AREA * REF_W * REF_H:
            why["large"] = why.get("large", 0) + 1
            continue
        samples = list(dict.fromkeys(str(ln["text"]) for ln in c))[:NOTICE_SAMPLES]
        out.append(
            {
                "kind": "notice",
                "x": rect[0],
                "y": rect[1],
                "w": rect[2] - rect[0],
                "h": rect[3] - rect[1],
                "score": float(st["bursts"] * st["distinct"]),
                "evidence": {"outside": True, **st, "samples": samples},
            }
        )
    out.sort(key=lambda p: (-p["score"], p["y"], p["x"]))
    out = out[: max(0, params.notice_max_regions - existing)]
    taken = {r.name for r in covered}
    k = 1
    for p in out:
        while (name := "extra_notice" if k == 1 else f"extra_notice{k}") in taken:
            k += 1
        p["name"] = name
        taken.add(name)
    logger.debug(
        "Outside notices: %d text lines outside %d covered regions, %d clusters, %d kept "
        "(dropped: %s), %d proposed",
        len(lines),
        len(covered),
        len(clusters),
        len(kept),
        ", ".join(f"{reason} {c}" for reason, c in sorted(why.items())) or "none",
        len(out),
    )
    return out


def notice_region(proposal: dict[str, Any]) -> dict[str, Any]:
    """The preset region entry of a :func:`notice_proposals` proposal: a ``notice``
    region on the det gate, every 3rd frame, detected at 0.6x for text of
    :data:`NOTICE_BIG_TEXT_H` px or more, marked ``"origin": "outside"``."""
    big = proposal["evidence"]["median_text_h"] >= NOTICE_BIG_TEXT_H
    r = Region(
        proposal["name"],
        proposal["x"],
        proposal["y"],
        proposal["w"],
        proposal["h"],
        "notice",
        gate="det",
        every=3,
        det_scale=0.6 if big else 1.0,
    )
    return asdict(r) | {"origin": "outside"}


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
    keep = keep or [p for p in proposals if p["kind"] not in ("readout", "popup")]
    out = []
    for p in keep[:max_regions]:
        kind = {"feed": "feed", "banner": "banner", "notice": "notice"}.get(p["kind"], "notice")
        out.append(_padded(p, kind, pad))
    return out


def _padded(p: dict[str, Any], kind: str, pad: int = 12) -> Region:
    """The :class:`Region` of proposal ``p``, padded by ``pad`` px and clipped to the frame."""
    x, y = max(0, p["x"] - pad), max(0, p["y"] - pad)
    w = min(REF_W - x, p["w"] + 2 * pad)
    h = min(REF_H - y, p["h"] + 2 * pad)
    return Region(p["name"], x, y, w, h, kind, gate="det", every=1 if kind == "feed" else 3)


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
        outside = p.get("evidence", {}).get("outside")
        c = (255, 0, 255) if outside else colors.get(p["kind"], (255, 255, 255))
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
    sources: list[Source],
    out_dir: Path,
    game: str,
    params: DiscoverParams | None = None,
    on_frame: Callable[[int, int], None] | None = None,
    covered: tuple[Region, ...] | None = None,
    outside: int = 0,
) -> dict[str, Any]:
    """Run discovery on ``sources`` and write the proposal into ``out_dir``.

    Args:
        sources: Sources of one game (decoded at the sampled times only).
        out_dir: Folder for ``frames/``, ``ocr.jsonl``, ``overlay.jpg`` and
            ``proposal.json`` (usually ``<project>/discovery/<game>``).
        game: Preset key the proposal is for.
        params: Sampling / clustering parameters.
        on_frame: Called with (frames OCR'd or read from the cache, frames planned)
            as the sampling goes on.
        covered: The regions of the game's existing preset (with the big-text
            region, :meth:`~cutpoint.events.regions.HudPreset.all_regions`); None
            when it has none.
        outside: How many ``"origin": "outside"`` regions that preset already has;
            they count against ``params.notice_max_regions``.

    Returns:
        The proposal: a preset dict (``game``, ``regions``, ``banner_phrases``,
        ``discovery`` stats, per-region evidence and the ids of the OCR records
        the run used, ``frames``, since ``ocr.jsonl`` keeps records of earlier
        runs). Extra notice regions where text shows outside every covered region
        (:func:`notice_proposals`, marked ``"origin": "outside"``) are added to
        ``regions`` when ``covered`` is None (the discovered regions and the
        big-text region count as covered), else listed in ``suggested_regions``
        only.
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
    frames = _ocr_frames(bursts, params, out_dir / "frames", out_dir / "ocr.jsonl", on_frame)
    proposals, maps = cluster_regions(frames, params)
    regions = proposals_to_regions(proposals)
    game_frames = gameplay_frames(frames, params)
    cover = covered if covered is not None else (*regions, bigtext_region())
    # score popups are not read, but their labels are no notices either
    cover += tuple(_padded(p, "notice") for p in proposals if p["kind"] == "popup")
    extra = notice_proposals(game_frames, cover, params, maps["static"], outside)
    extra_regions = [notice_region(p) for p in extra]
    proposals += extra
    info = probe(sources[0].io_path) if sources else None
    write_overlay(
        game_frames,
        out_dir / "frames",
        maps,
        proposals,
        out_dir / "overlay.jpg",
    )
    names = {r.name for r in regions if r.kind == "feed"}
    icon = any(p["evidence"].get("icon_feed") for p in proposals if p.get("name") in names)
    preset = {
        "game": game,
        "reference": [REF_W, REF_H],
        "source": "discovered",
        "regions": [asdict(r) for r in regions] + (extra_regions if covered is None else []),
        "banner_phrases": [],
        # an icon feed's other rows ("Kip has left the game") are notices
        **({"feed": {"row_notices": True}} if icon else {}),
        "discovery": {
            **maps["stats"],
            "params": asdict(params),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "source_resolution": [info.width, info.height] if info else None,
            "proposals": proposals,
            "frames": [f["id"] for f in frames],  # this run's OCR records
        },
    }
    if covered is not None:
        preset["suggested_regions"] = extra_regions
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
    if extra:  # rectangles and counts only: OCR text stays in proposal.json
        logger.info(
            "Discovery: text shows outside the %s regions; %s %d extra notice region(s): %s",
            "preset's" if covered is not None else "discovered",
            "suggesting" if covered is not None else "adding",
            len(extra),
            ", ".join(
                f"{p['name']}@{p['x']},{p['y']} {p['w']}x{p['h']} "
                f"({p['evidence']['bursts']} bursts, {p['evidence']['frames']} frames)"
                for p in extra
            ),
        )
    return preset


__all__ = [
    "DiscoverParams",
    "boxes_in",
    "cluster_regions",
    "discover_regions",
    "gameplay_frames",
    "notice_proposals",
    "notice_region",
    "read_ocr_samples",
]
