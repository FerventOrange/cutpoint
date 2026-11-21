"""The OCR read and its interpretation (split from ``killfeed.py``, #1085 A347).

:class:`OcrReader` OCRs the HUD regions of each decoded frame into :class:`FrameRec`
records; :class:`OcrInterpretation` (through :class:`Interpreter` per region) turns them into
events.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cutpoint.events.feed_parser import (
    FeedParser,
    _alnum_len,
    _fold,
    _rows,
    by_centre,
    fuzzy_contains,
    norm,
    similar,
)
from cutpoint.events.menu_rules import (
    MENU,
    StillRuns,
    _grey,
    _log_menu_frames,
    _still_rect,
    _thumb_bgr,
    _union,
    feed_stacks,
    menu_rules,
    staying_spans,
    still_stretches,
)
from cutpoint.events.menu_rules import logger as menu_logger
from cutpoint.events.menu_spans import merge_menu_frames
from cutpoint.events.ocr_engine import OcrEngine
from cutpoint.events.regions import (
    REF_H,
    HudPreset,
    Region,
    letters,
    marker_seen,
    ref_factors,
    scaled,
)
from cutpoint.events.score_readout import ScoreTracker
from cutpoint.events.text import notice_letters
from cutpoint.events.tracker import (
    FUZZY,
    KILL_ROW_S,
    LINE_ENTRIES,
    PERSIST_S,
    REPLAY_MIN_LETTERS,
    EventTracker,
    collapse_banner_bursts,
    learn_announcements,
    mark_replay_kills,
    merge_kill_reads,
    replay_spans,
)

if TYPE_CHECKING:
    from cutpoint.events.killfeed import OcrParams

logger = logging.getLogger(__name__)

MIN_TEXT_SCORE = 0.6  # recognised text below this score is dropped

# Chat overlays (Twitch VODs): a discovered "feed" region that shows this many distinct text
# lines per hour of footage is chat, not a kill feed. Measured on real footage: chat 9000-40000
# lines/h, kill feeds and game logs 30-2200. Neither the share of frames with text nor colons
# or name counts separate them.
CHAT_LINES_PER_HOUR = 6000  # at or above: label "chat" (no events)
CHAT_UNSURE_LINES_PER_HOUR = 3000  # from here up to CHAT_LINES_PER_HOUR: stays "feed", warned
# A short sample says too little (a menu in the region can read as many lines in minutes):
# at least 10 minutes of footage.
CHAT_MIN_FOOTAGE_S = 600.0
CHAT_MIN_CONF = 0.6  # text boxes read less surely are not counted
CHAT_MIN_CHARS = 3  # nor those with fewer letters and digits
# Feed rows read as notices (``row_notices``) that stay longer than this are no feed lines but
# text that stays in the feed's area (a lobby line: "We're estimating your chances to win..."
# for 49 s in one VOD); feed lines scroll out within seconds (kill lines and notices
# there: up to 10 s).
ROW_NOTICE_MAX_S = 20.0
# Two words that come back this often with neither ever seen with another name are a label
# ("Target | Marked"), not two players: a replay shows a kill twice at most
LABEL_REPEATS = 3
# A system line of the feed ("friendly fire is now active") read as a kill row with the names
# cut in the wrong place. The same line read as a notice recurs, a kill's two names do not read
# like it: a notice text that comes back NOTICE_LINE_REPEATS times in one region (reads at least
# NOTICE_LINE_APART_S apart, each similar to the first, :data:`NOTICE_LINE_SIMILAR`) is a system
# line, and a kill in that region whose two names joined read like it (at least
# NOTICE_LINE_MIN_CHARS letters and digits) is that line. Measured on gameplay footage: 12 of 270
# kill rows were such a line; no real kill row was. A notice that is a kill's own row (read as a
# notice when the icon gap was missed) does not count.
NOTICE_LINE_REPEATS = 5
NOTICE_LINE_APART_S = 10.0
NOTICE_LINE_SIMILAR = 0.7
NOTICE_LINE_MIN_CHARS = 12


def _two_hues(e: dict[str, Any]) -> bool:
    """Whether a kill line's names have two team colours, which labels and menu rows do not."""
    hues = {e.get("killer_hue"), e.get("victim_hue")}
    return len(hues) == 2 and not hues & {None, "white"}


class Interpreter:
    """Turns OCR results of one region into tracked events."""

    def __init__(self, params: OcrParams, scale: float, preset: HudPreset | None = None) -> None:
        self.params = params
        self.scale = scale
        self.preset = preset
        self.phrases = preset.phrases if preset else ()
        self.tags = tuple(
            dict.fromkeys([*params.player_tags, *(preset.player_tags if preset else ())])
        )
        self.tracker = EventTracker(PERSIST_S, FUZZY)
        # short big-text lines with digits ("4 VS 5"), which make no event: tracked apart for
        # :func:`learn_announcements` only, so the menu test and the kill and feed rules never
        # see them
        self.digits = EventTracker(PERSIST_S, FUZZY)
        self.feed = FeedParser(preset.feed if preset else {}, self.tags, scale)

    def drop_labels(self) -> None:
        """Forget the icon kills that are no kills by how they behave: a line that stays
        unbroken longer than :data:`ROW_NOTICE_MAX_S`, and two words that come back
        :data:`LABEL_REPEATS` times with neither ever seen with another name. Lines with two
        team colours stay (:func:`_two_hues`): two kills of one pair can follow each other."""
        tracker = self.tracker
        lines = [e for e in tracker.events if e["type"] == "kill" and e.get("entry") == "icon"]
        # The name test of a line depends on its (killer, victim) only, so it is made once per
        # distinct pair, and each name comparison once (#1085, A703: it was every line with
        # every other, twice). Lines are not bucketed by killer: a misread killer still matches.
        counts = Counter((e["killer"], e["victim"]) for e in lines)
        same_memo: dict[tuple[str | None, str | None], bool] = {}
        side_memo: dict[tuple[str | None, str | None], bool] = {}
        verdicts: dict[tuple[str, str], bool] = {}

        def same(a: str | None, b: str | None) -> bool:
            if (a, b) not in same_memo:
                same_memo[a, b] = tracker._nested(a, b) or tracker._misread(a, b)
            return same_memo[a, b]

        def same_side(a: str | None, b: str | None) -> bool:
            """One side of :meth:`EventTracker._same_line`."""
            if (a, b) not in side_memo:
                side_memo[a, b] = same(a, b) or similar(a or "", b or "") >= tracker.fuzzy
            return side_memo[a, b]

        def names_label(p: tuple[str, str]) -> bool:
            if p not in verdicts:
                # :meth:`EventTracker._same_line` of two lines with these names
                repeats = {q for q in counts if same_side(p[0], q[0]) and same_side(p[1], q[1])}
                verdicts[p] = sum(counts[q] for q in repeats) >= LABEL_REPEATS and not any(
                    same(n, o) for q in counts if q not in repeats for n in p for o in q
                )
            return verdicts[p]

        def label(e: dict[str, Any]) -> bool:
            if _two_hues(e):
                return False
            runs = _runs(e["_times"], MENU.static_feed_guard_s)
            if max(run[-1] - run[0] for run in runs) > ROW_NOTICE_MAX_S:
                return True
            return names_label((e["killer"], e["victim"]))

        gone = {id(e) for e in lines if label(e)}
        tracker.events = [e for e in tracker.events if id(e) not in gone]

    def drop_notice_kills(self) -> None:
        """Forget the kills that are a system line read as a kill row (see
        :data:`NOTICE_LINE_REPEATS`)."""
        tracker = self.tracker
        lines: dict[Any, list[list[Any]]] = {}  # region -> [first text, texts, read times]
        kills = [e for e in tracker.events if e["type"] == "kill"]

        def own_row(e: dict[str, Any], text: str) -> bool:
            """Whether the notice is the row of a kill in its region, read without the icon gap."""
            return any(
                k.get("region") == e.get("region")
                and e["t"] - KILL_ROW_S <= k["t"] <= e["end"] + KILL_ROW_S
                and difflib.SequenceMatcher(
                    None, _fold(k.get("killer") or "") + _fold(k.get("victim") or ""), text
                ).ratio()
                >= NOTICE_LINE_SIMILAR
                for k in kills
            )

        for e in sorted((e for e in tracker.events if e["type"] == "notice"), key=lambda e: e["t"]):
            text = _fold(e.get("text") or "")
            if len(text) < NOTICE_LINE_MIN_CHARS or own_row(e, text):
                continue
            if e.get("unknown") and e.get("sightings", 1) < 2:
                continue
            for line in lines.setdefault(e.get("region"), []):
                if difflib.SequenceMatcher(None, line[0], text).ratio() >= NOTICE_LINE_SIMILAR:
                    line[1].add(text)
                    if e["t"] - line[2][-1] >= NOTICE_LINE_APART_S:
                        line[2].append(e["t"])
                    break
            else:
                lines[e.get("region")].append([text, {text}, [e["t"]]])
        recurring = {
            region: {t for line in found if len(line[2]) >= NOTICE_LINE_REPEATS for t in line[1]}
            for region, found in lines.items()
        }

        def system_line(e: dict[str, Any]) -> bool:
            names = _fold(e.get("killer") or "") + _fold(e.get("victim") or "")
            return len(names) >= NOTICE_LINE_MIN_CHARS and any(
                difflib.SequenceMatcher(None, names, t).ratio() >= NOTICE_LINE_SIMILAR
                for t in recurring.get(e.get("region"), ())
            )

        tracker.events = [e for e in tracker.events if not (e["type"] == "kill" and system_line(e))]

    def _phrases(self, t: float, region: Region, text: str, conf: float) -> bool:
        """Phrase matches, keeping only the longest of nested matches
        ("YOU HAVE BEEN DESTROYED", not also "DESTROYED"); the text after the
        phrase is kept as ``detail`` ("by -XMPL- Pinecone4711 (Strv m/40L)")."""
        hits = [ph for ph in self.phrases if fuzzy_contains(text, ph, FUZZY)]
        hits = [ph for ph in hits if not any(ph != o and norm(ph) in norm(o) for o in hits)]
        for ph in hits:
            last = norm(ph).split()[-1]
            i = text.upper().find(last)
            detail = text[i + len(last) :].strip(" :.-")[:80] if i >= 0 else ""
            self.tracker.see(
                t,
                "banner",
                ph,
                text=ph,
                raw=text[:120],
                detail=detail,
                confidence=round(conf, 3),
                region=region.name,
            )
        return bool(hits)

    def handle(
        self, t: float, region: Region, img: np.ndarray, items: list[tuple[list, str, float]]
    ) -> None:
        """Interpret OCR ``items`` of ``region`` (crop ``img``) at time ``t``: feeds,
        banners, notices and big text. A "score" region makes no text event: its reads go to
        :class:`ScoreTracker` only."""
        if not items or region.kind == "score":
            return
        if region.kind == "replay":
            # a replay or killcam marker the vision HUD map pointed at: text there means a
            # replay is on screen (one event per stretch; what it says does not matter)
            # each OCR item on its own: a feed row that reaches into the box is no marker
            # (or, when none is, the items of one row joined: a marker read in parts)
            cap = 2 * len(letters(region.marker))

            def marked(text: str) -> bool:
                return sum(ch.isalpha() for ch in text) >= REPLAY_MIN_LETTERS and (
                    not region.marker
                    or (len(letters(text)) <= cap and marker_seen(text, region.marker))
                )

            hits = [i[1] for i in items if marked(i[1])]
            if not hits and len(items) > 1:
                joined = (" ".join(i[1] for i in row) for row in _rows(items, 12 * self.scale))
                hits = [text for text in joined if marked(text)]
            line = " ".join(hits).strip()
            if line:
                self.tracker.see(
                    t,
                    "replay",
                    "replay",
                    text=line[:40],
                    confidence=round(float(np.mean([i[2] for i in items])), 3),
                    region=region.name,
                )
            return
        if region.kind == "feed":
            for ev in self.feed.parse(img, items):
                # exclusive: identical rows on screen at once are separate kills
                self.tracker.see(
                    t, ev.pop("type"), ev.pop("key"), exclusive=True, region=region.name, **ev
                )
            return
        min_h = region.min_text_h * self.scale
        rows = _rows(items, 12 * self.scale)
        for row in rows:
            h = float(np.median([max(p[1] for p in i[0]) - min(p[1] for p in i[0]) for i in row]))
            if region.kind in ("bigtext", "banner") and h < min_h:
                continue
            line = " ".join(i[1] for i in row).strip()
            c = float(np.mean([i[2] for i in row]))
            if self._phrases(t, region, line, c):
                continue
            if region.kind == "bigtext" and h < 1.4 * min_h:
                continue  # unknown text must be really big to count as a banner
            etype = "notice" if region.kind == "notice" else "bigtext"
            tracker = self.tracker
            if not notice_letters(line):
                if etype == "notice" or not any(ch.isdigit() for ch in line):
                    continue  # ranges, reticle marks; numbers in a notice are never learned
                tracker = self.digits  # numbers: no event unless learned as an announcement
            tracker.see(
                t,
                etype,
                norm(line),
                text=line[:120],
                confidence=round(c, 3),
                region=region.name,
                text_h=round(h / self.scale),
                unknown=True,
            )


def _runs(times: list[float], gap: float) -> list[list[float]]:
    """The sorted ``times`` in runs without a step longer than ``gap``."""
    runs: list[list[float]] = []
    for t in sorted(times):
        if runs and t - runs[-1][-1] <= gap:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def group_rows(
    boxes: list[list], min_h: float, tol: float
) -> list[tuple[float, float, float, float, float]]:
    """Group detected text boxes into rows ``(cy, x0, x1, y0, y1)``, top to bottom.

    Boxes shorter than ``min_h`` are ignored; boxes whose vertical centres are
    within ``tol`` of the previous row's centre join that row.
    """
    kept = [b for b in boxes if not (min_h and _box_h(b) < min_h)]
    return [
        (
            _box_cy(row[0]),
            min(min(p[0] for p in b) for b in row),
            max(max(p[0] for p in b) for b in row),
            min(min(p[1] for p in b) for b in row),
            max(max(p[1] for p in b) for b in row),
        )
        for row in by_centre(kept, _box_cy, tol, order=lambda b: sum(p[1] for p in b))
    ]


def _box_h(b: list) -> Any:
    return max(p[1] for p in b) - min(p[1] for p in b)


def _box_cy(b: list) -> Any:
    return (min(p[1] for p in b) + max(p[1] for p in b)) / 2


# --------------------------------------------------------------------------- reader


# Region kinds whose interpretation looks at pixels: only their reads keep a band.
PIXEL_KINDS = ("feed",)

# Det-gate skip (``OcrParams.det_skip``, #807): a det-gated region whose grey crop, scaled
# down DET_SKIP_SCALE times, differs from the one of its last det run by less than
# DET_SKIP_MAX_DIFF (0-255) in every DET_SKIP_TILE px tile (mean abs difference) keeps that
# run's outcome without a det call. Measured: a third and a tenth of the det calls on two
# clips, no event lost.
DET_SKIP_SCALE = 4
DET_SKIP_TILE = 8
DET_SKIP_MAX_DIFF = 8.0


def _skip_image(crop: np.ndarray) -> np.ndarray:
    """The grey crop scaled down :data:`DET_SKIP_SCALE` times (uint8), the det skip compares."""
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    size = (max(1, w // DET_SKIP_SCALE), max(1, h // DET_SKIP_SCALE))
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA)


def _tile_diff(a: np.ndarray, b: np.ndarray) -> float:
    """The largest mean abs difference of ``a`` and ``b`` over :data:`DET_SKIP_TILE` px tiles
    (the tiles at the right and bottom edges may be smaller)."""
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    ys = np.arange(0, d.shape[0], DET_SKIP_TILE)
    xs = np.arange(0, d.shape[1], DET_SKIP_TILE)
    sums = np.add.reduceat(np.add.reduceat(d, ys, axis=0), xs, axis=1)
    counts = np.outer(np.diff(ys, append=d.shape[0]), np.diff(xs, append=d.shape[1]))
    return float((sums / counts).max())


@dataclass
class FrameRec:
    """One decoded frame as :class:`OcrReader` saw it: all :class:`OcrInterpretation`
    needs, and what the raw OCR cache keeps (:mod:`cutpoint.events.ocr_raw`).

    ``regions`` has a record per region the reader looked at: ``{"g": "det"}`` (a
    "det" gate call; the layout did not change and has no text rows), ``{"g":
    "keep"}`` (the layout did not change: the text is still on screen) or ``{"g":
    "read", "items": [[box, text, score], ...]}`` (an OCR call). Records of "det"
    gate regions other than feeds carry the gate's ``boxes`` (reference px ``[x0,
    y0, x1, y1]``, for the menu test); a "det" or "keep" record the det skip made
    without a det call (``OcrParams.det_skip``) has ``"s": 1`` and the boxes of the
    region's last det call; reads in regions whose kind looks at pixels
    (:data:`PIXEL_KINDS`) carry ``rows``, the crop rows ``[a, b]`` their
    boxes cover, whose pixels are stacked in ``bands``. A region skipped by its
    ``every`` has no record.
    ``thumb`` is the :func:`_thumb_bgr` thumbnail of the :func:`_still_rect` area (None
    without "det" gates).
    """

    n: int
    t: float
    thumb: np.ndarray | None = None
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    bands: dict[str, np.ndarray] = field(default_factory=dict)


def _plain(v: Any) -> Any:
    """A numpy number as a Python one (JSON; the same arithmetic live and replayed)."""
    return v.item() if hasattr(v, "item") else v


def ref_box(box: list, ref: tuple[float, float, float, float]) -> list[Any]:
    """An OCR box of a crop as ``[x0, y0, x1, y1]`` in reference px (#1085 A341); ``ref``:
    the crop's left and top in frame px and the frame-to-reference factors."""
    rx, ry, sx, sy = ref
    return [
        _plain((rx + min(p[0] for p in box)) * sx),
        _plain((ry + min(p[1] for p in box)) * sy),
        _plain((rx + max(p[0] for p in box)) * sx),
        _plain((ry + max(p[1] for p in box)) * sy),
    ]


def _band_rows(items: list[list], h: int) -> list[list[int]]:
    """The rows of a crop of height ``h`` that the boxes of OCR ``items`` cover,
    merged: all the rows the colour rules read (:func:`_box_crop` and
    :func:`_merge_adjacent`, whose boxes keep the rows of their parts)."""
    spans = []
    for box, _, _ in items:
        ys = [int(p[1]) for p in box]
        a, b, _ = slice(max(0, min(ys)), max(ys)).indices(h)
        if b > a:
            spans.append([a, b])
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _band_image(
    shape: tuple[int, ...], rows: list[list[int]], band: np.ndarray | None
) -> np.ndarray:
    """A crop of ``shape``: the pixels of ``band`` at ``rows``, black elsewhere."""
    img = np.zeros(shape, np.uint8)
    off = 0
    for a, b in rows:
        img[a:b] = band[off : off + b - a]
        off += b - a
    return img


def _crop_shape(
    rect: tuple[int, int, int, int], union: tuple[int, int, int, int]
) -> tuple[int, ...]:
    """Shape of a region's crop of the decoded ``union`` (cut where it ends)."""
    x, y, w, h = rect
    ux, uy, uw, uh = union
    return len(range(uh)[y - uy : y - uy + h]), len(range(uw)[x - ux : x - ux + w]), 3


class OcrReader:
    """The gates and OCR calls of :func:`detect_ocr_events`: decoded frames in,
    :class:`FrameRec` records out. Its state (each region's last layout, rows and
    mask) goes into every raw OCR cache chunk, so a read resumes."""

    def __init__(
        self,
        regions: tuple[Region, ...],
        rects: dict[str, tuple[int, int, int, int]],
        union: tuple[int, int, int, int],
        engine: OcrEngine,
        params: OcrParams,
        scale: float,
        newest: str,
        size: tuple[int, int],
    ) -> None:
        self.regions = regions
        self.rects = rects
        self.union = union
        self.engine = engine
        self.params = params
        self.scale = scale
        self.newest = -1 if newest == "bottom" else 0
        self.sx, self.sy = ref_factors(*size)
        self.thumbs = any(r.gate == "det" for r in regions)  # the static-screen test
        x, y, w, h = _still_rect(regions, rects)
        self.still = (slice(y - union[1], y - union[1] + h), slice(x - union[0], x - union[0] + w))
        self.last_layout: dict[str, list[tuple[int, ...]] | None] = {r.name: None for r in regions}
        self.last_rows: dict[str, list[tuple[float, float, float, float, float]]] = {}
        # det skip: per det-gated region the image of its last det run and that run's boxes
        self.det_skip = params.det_skip
        self.anchor: dict[str, np.ndarray] = {}
        self.last_boxes: dict[str, list[list[float]]] = {}

    def _det_changed(self, r: Region, crop: np.ndarray, rec: dict[str, Any]) -> bool:
        """Text-layout gate: True when the detected text *rows* changed.

        Boxes are grouped into rows by vertical centre; a row is (centre y,
        left x, right x) quantised to ~8 reference px. Detection jitter splits
        and merges boxes inside a row, so comparing rows instead of boxes
        keeps recognition to real changes (a new feed entry, a scroll, a new
        message).
        """
        boxes = self.engine.detect(crop, r.det_scale)
        if r.kind != "feed":  # a full feed is no menu
            rx, ry, _, _ = self.rects[r.name]
            ref = (rx, ry, self.sx, self.sy)
            rec["boxes"] = [ref_box(b, ref) for b in boxes]
        rows = group_rows(boxes, r.min_text_h * self.scale, 10 * self.scale)
        q = 8 * self.scale
        layout = [tuple(round(v / q) for v in (cy, x0, x1)) for cy, x0, x1, _, _ in rows]
        prev, self.last_layout[r.name] = self.last_layout[r.name], layout
        self.last_rows[r.name] = rows
        if not layout:
            return False
        if r.kind == "feed":
            # Feeds add entries at one end (by default the bottom) and scroll or
            # fade the rest, which were already read: recognise only when a row
            # was added or the newest row changed.
            if not prev or len(layout) > len(prev):
                return True
            new = self.newest
            return max(abs(a - b) for a, b in zip(prev[new], layout[new], strict=True)) > 1
        if prev is None or len(prev) != len(layout):
            return True
        return any(
            max(abs(a - b) for a, b in zip(pa, pb, strict=True)) > 1
            for pa, pb in zip(prev, layout, strict=True)
        )

    def _look(
        self, r: Region, crop: np.ndarray
    ) -> tuple[dict[str, Any], list[tuple[float, float, float, float]] | None, list | None]:
        """The gate of region ``r`` and its OCR calls but the row recognition: ``(record,
        rows, items)``: a det-gated region to read has its ``rows`` (``y0, x0, y1, x1``), a
        read-gated one its ``items``; neither when the layout did not change."""
        if r.gate == "det":
            g: dict[str, Any] = {}
            if self.det_skip:
                small = _skip_image(crop)
                anchor = self.anchor.get(r.name)
                if anchor is not None and _tile_diff(small, anchor) < DET_SKIP_MAX_DIFF:
                    # unchanged since the last det run: its outcome, without a det call
                    g["g"] = "keep" if self.last_layout[r.name] else "det"
                    g["s"] = 1
                    if r.name in self.last_boxes:
                        g["boxes"] = list(self.last_boxes[r.name])
                    return g, None, None
                self.anchor[r.name] = small
            changed = self._det_changed(r, crop, g)
            if self.det_skip and "boxes" in g:
                self.last_boxes[r.name] = g["boxes"]
            if not changed:
                g["g"] = "keep" if self.last_layout[r.name] else "det"
                return g, None, None
            rows = self.last_rows[r.name]
            if r.det_scale < 1.0:
                # the cheap gate's boxes clip the ends of rows: re-detect at
                # full resolution for exact row boxes (~35 ms)
                rows = group_rows(
                    self.engine.detect(crop), r.min_text_h * self.scale, 10 * self.scale
                )
            return g, [(y0, x0, y1, x1) for _, x0, x1, y0, y1 in rows], None
        return {}, None, self._full_read(r, crop)  # the "read" gate

    def _full_read(self, r: Region, crop: np.ndarray) -> list[tuple[list, str, float]]:
        """Detection and recognition on the crop, enlarged first when ``r.upscale`` asks for
        more than the decoded size (``upscale`` x the reference size; ``upscale`` 1 reads the
        crop as decoded, whatever the source's size); boxes in crop px."""
        f = r.upscale / self.scale
        if r.upscale == 1.0 or f <= 1.0:
            return self.engine(crop, MIN_TEXT_SCORE)
        import cv2  # noqa: PLC0415

        big = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        return [
            ([[p[0] / f, p[1] / f] for p in box], text, score)
            for box, text, score in self.engine(big, MIN_TEXT_SCORE)
        ]

    def read(self, n: int, t: float, frame: np.ndarray) -> tuple[FrameRec, dict[str, np.ndarray]]:
        """The record of frame ``n`` (time ``t``) and the crops of the regions read."""
        rec = FrameRec(n, t, _thumb_bgr(frame[self.still]) if self.thumbs else None)
        crops: dict[str, np.ndarray] = {}
        ux, uy, _, _ = self.union
        due = [r for r in self.regions if r.every <= 1 or not n % r.every]
        cut = {}
        for r in due:
            x, y, w, h = self.rects[r.name]
            cut[r.name] = frame[y - uy : y - uy + h, x - ux : x - ux + w]
        due = [r for r in due if cut[r.name].size]  # a crop clipped to nothing has no text
        # the regions' gates at once (each region's state is its own), then the rows in order;
        # the rows of new regions are keyed in region order, as the state is saved as is
        for r in due:
            if r.gate == "det":
                self.last_rows.setdefault(r.name, [])
        each = getattr(self.engine, "map", None) or (lambda fn, xs: [fn(x) for x in xs])
        looks = each(lambda r: self._look(r, cut[r.name]), due)
        for r, (g, text_rows, items) in zip(due, looks, strict=True):
            rec.regions[r.name] = g
            crop = cut[r.name]
            if text_rows is not None:
                items = self.engine.read_rows(crop, text_rows, MIN_TEXT_SCORE, pad=6 * self.scale)
            elif items is None:
                continue
            g["g"] = "read"
            g["items"] = [
                [[[_plain(v) for v in p] for p in box], str(text), float(score)]
                for box, text, score in items
            ]
            if r.kind in PIXEL_KINDS and (rows := _band_rows(g["items"], crop.shape[0])):
                g["rows"] = rows
                rec.bands[r.name] = np.concatenate([crop[a:b] for a, b in rows])
            crops[r.name] = crop
        return rec, crops

    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """The reader's state: JSON (layouts, rows; with the det skip its last boxes) and
        arrays (with the det skip its images)."""
        arrays: dict[str, np.ndarray] = {}
        state: dict[str, Any] = {"layout": self.last_layout, "rows": self.last_rows}
        if self.det_skip:
            state["boxes"] = self.last_boxes
            arrays = {
                f"anchor{i}": a
                for i, r in enumerate(self.regions)
                if (a := self.anchor.get(r.name)) is not None
            }
        return state, arrays

    def restore(self, state: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        """Continue from a :meth:`state`."""
        layout = state["layout"]
        self.last_layout = {
            r.name: None if layout.get(r.name) is None else [tuple(v) for v in layout[r.name]]
            for r in self.regions
        }
        self.last_rows = {k: [tuple(v) for v in rows] for k, rows in state["rows"].items()}
        self.last_boxes = {
            k: [list(b) for b in boxes] for k, boxes in state.get("boxes", {}).items()
        }
        self.anchor = {
            r.name: a
            for i, r in enumerate(self.regions)
            if (a := arrays.get(f"anchor{i}")) is not None
        }


# --------------------------------------------------------------------------- interpretation


def _chat_lines(items: Iterable[tuple[Any, str, float]]) -> set[str]:
    """The distinct text lines of one read, for :func:`chat_label`: lower-case letters
    and digits (any script) and single spaces of the sure enough, long enough text boxes."""
    out = set()
    for _box, text, score in items:
        line = " ".join(re.sub(r"[^\w ]|_", "", text.casefold()).split())
        if score >= CHAT_MIN_CONF and _alnum_len(line) >= CHAT_MIN_CHARS:
            out.add(line)
    return out


def chat_label(lines: int, hours: float) -> tuple[str, float]:
    """How a feed region reads, from the ``lines`` distinct text lines it showed in
    ``hours`` of footage: ("chat" | "unsure" | "short" | "feed", lines per hour). A chat
    overlay shows far more different lines than a kill feed. Footage under
    :data:`CHAT_MIN_FOOTAGE_S` is too short to tell: "short" at the chat rate, else "feed".
    """
    rate = lines / hours if hours > 0 else 0.0
    if hours * 3600 < CHAT_MIN_FOOTAGE_S:
        return ("short" if rate >= CHAT_LINES_PER_HOUR else "feed"), rate
    if rate >= CHAT_LINES_PER_HOUR:
        return "chat", rate
    if rate >= CHAT_UNSURE_LINES_PER_HOUR:
        return "unsure", rate
    return "feed", rate


class OcrInterpretation:
    """The interpretation of :func:`detect_ocr_events`: :class:`FrameRec` records in,
    events out (counts go into ``stats``). A live read gives it each record as the
    reader makes it, so a replay of the raw OCR cache is the same by construction."""

    def __init__(
        self,
        src_name: str,
        params: OcrParams,
        preset: HudPreset | None,
        regions: tuple[Region, ...],
        size: tuple[int, int],
        stats: dict[str, Any],
    ) -> None:
        width, height = size
        self.src_name = src_name
        self.regions = regions
        self.interp = Interpreter(params, height / REF_H, preset)
        rects = {r.name: scaled(r, width, height) for r in regions}
        union = _union(list(rects.values()))
        self.shapes = {name: _crop_shape(rect, union) for name, rect in rects.items()}
        self.stats = stats
        stats |= {"frames": 0, "ocr_calls": {r.name: 0 for r in regions}}
        det = any(r.gate == "det" for r in regions)
        if det:
            stats["det_calls"] = {r.name: 0 for r in regions if r.gate == "det"}
            if params.det_skip:  # det calls the det skip saved (not in det_calls)
                stats["det_skipped"] = {r.name: 0 for r in regions if r.gate == "det"}
        # menu test: the det boxes of every region in reference px, per frame
        self.menus = preset is not None and det
        self.det_boxes: dict[str, list[list[float]]] = {}
        self.frame_boxes: list[dict[str, Any]] = []
        # static screens (see MENU.static_thumb): only when the HUD area sees enough of the frame
        _, _, still_w, still_h = _still_rect(regions, rects)
        area = still_w * still_h / max(1, width * height)
        self.still_runs = (
            StillRuns(params.fps) if self.menus and area >= MENU.static_min_area else None
        )
        if self.menus:
            stats["static_area"] = round(area, 2)
        self.region_events: dict[str, list[dict[str, Any]]] = {}
        self.scores = ScoreTracker()  # the "score" regions' reads
        # the "score" regions' rectangles in frame px: a text line centred in one is a counter
        # read by a text region too, not text
        self.score_zones = [
            (x, y, x + w, y + h)
            for r in regions
            if r.kind == "score"
            for x, y, w, h in [rects[r.name]]
        ]
        # distinct text lines read per feed region (:func:`chat_label`); only a discovered
        # preset's regions can be labelled chat, a hand-made preset's feed is the user's word
        self.lines: dict[str, set[str]] = {r.name: set() for r in regions if r.kind == "feed"}
        self.discovered = preset is not None and preset.source == "discovered"
        # a "row_notices" feed's notice rows count in the menu test (a menu or loading
        # screen's text there), in reference px
        self.feed_notices = bool(preset and preset.feed.get("row_notices"))
        self.ref = {n: (x, y, *ref_factors(width, height)) for n, (x, y, _, _) in rects.items()}

    def _outside_scores(
        self, items: list[tuple[list, str, float]], ref: tuple[float, float, float, float]
    ) -> list[tuple[list, str, float]]:
        """``items`` of a text region (crop at ``ref``) without the lines whose box centre lies
        in a "score" region."""
        rx, ry = ref[0], ref[1]

        def inside(box: list) -> bool:
            cx = rx + (min(p[0] for p in box) + max(p[0] for p in box)) / 2
            cy = ry + (min(p[1] for p in box) + max(p[1] for p in box)) / 2
            return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in self.score_zones)

        return [it for it in items if not inside(it[0])]

    def add(self, rec: FrameRec) -> list[str]:
        """Interpret one frame; returns the names of the regions whose reading made a
        new event."""
        interp, tracker, stats = self.interp, self.interp.tracker, self.stats
        stats["frames"] += 1
        new = []
        for r in self.regions:
            g = rec.regions.get(r.name)
            if g is None:
                continue
            if r.gate == "det":
                stats["det_skipped" if g.get("s") else "det_calls"][r.name] += 1
                if self.menus and r.kind != "feed":
                    self.det_boxes[r.name] = g.get("boxes", [])
                elif self.menus and self.feed_notices and g["g"] == "det":
                    self.det_boxes[r.name] = []  # no text rows
                if g["g"] == "keep":  # same text still on screen
                    tracker.keep_alive(self.region_events.get(r.name, []), rec.t)
                if g["g"] != "read":
                    continue
            stats["ocr_calls"][r.name] += 1
            img = _band_image(self.shapes[r.name], g.get("rows", []), rec.bands.get(r.name))
            items = [(box, text, score) for box, text, score in g["items"]]
            if r.kind == "score":
                self.scores.see(rec.t, r.name, items, self.ref[r.name])
                continue
            if self.score_zones:
                items = self._outside_scores(items, self.ref[r.name])
            if r.name in self.lines:
                self.lines[r.name].update(_chat_lines(items))
            n_before = len(tracker.events)
            if r.gate == "det":
                tracker.reset_touched()
                interp.digits.reset_touched()
                interp.feed.notice_boxes = []
                interp.handle(rec.t, r, img, items)
                self.region_events[r.name] = [*tracker.touched, *interp.digits.touched]
                if self.menus and self.feed_notices and r.kind == "feed":
                    ref = self.ref[r.name]
                    self.det_boxes[r.name] = [ref_box(b, ref) for b in interp.feed.notice_boxes]
            else:
                interp.handle(rec.t, r, img, items)
            if len(tracker.events) > n_before:
                new.append(r.name)
        if self.menus:
            # regions overlap (notices inside the big-text area): one box per ~12 px spot
            spots = {
                (round((a + c) / 24), round((b + d) / 24)): (a, b, c, d)
                for bs in self.det_boxes.values()
                for a, b, c, d in bs
            }
            self.frame_boxes.append(
                {
                    "id": rec.n,
                    "t": rec.t,
                    "source": self.src_name,
                    "boxes": [
                        {"x0": a, "y0": b, "x1": c, "y1": d} for a, b, c, d in spots.values()
                    ],
                }
            )
            if self.still_runs is not None and rec.thumb is not None:
                self.frame_boxes[-1]["still"] = self.still_runs.add(rec.t, _grey(rec.thumb))
        return new

    def _drop_chat(self) -> None:
        """Label the feed regions (:func:`chat_label`) from the lines the whole pass read,
        and forget the events of the ones that are chat. The raw text stays in the raw OCR
        cache. ``stats["feed_regions"]``: per feed region its ``label``, ``lines_per_hour``,
        ``lines``, ``hours`` and the ``dropped`` event counts by type."""
        tracker, stats = self.interp.tracker, self.stats
        hours = stats["frames"] / self.interp.params.fps / 3600
        report: dict[str, dict[str, Any]] = {}
        chat: set[str] = set()
        for name, lines in self.lines.items():
            label, rate = chat_label(len(lines), hours)
            info: dict[str, Any] = {
                "label": label,
                "lines_per_hour": round(rate),
                "lines": len(lines),
                "hours": round(hours, 3),
            }
            if label == "chat" and not self.discovered:
                logger.warning(
                    "%s: region %r reads like chat (%.0f lines/h) but the preset was not "
                    "discovered; kept as a feed",
                    self.src_name,
                    name,
                    rate,
                )
                info["label"] = "feed"
                info["chat_like"] = True
            elif label == "chat":
                chat.add(name)
                logger.info(
                    "%s: region %r reads like stream chat (%.0f lines/h): no feed events",
                    self.src_name,
                    name,
                    rate,
                )
            elif label == "short":
                info["label"] = "feed"
                info["too_short"] = True
                logger.warning(
                    "%s: region %r reads %.0f lines/h, like chat, but the footage is too "
                    "short to tell; kept as a feed",
                    self.src_name,
                    name,
                    rate,
                )
            elif label == "unsure":
                info["label"] = "feed"
                info["unsure"] = True
                logger.warning(
                    "%s: region %r may be stream chat (%.0f lines/h); kept as a feed",
                    self.src_name,
                    name,
                    rate,
                )
            report[name] = info | {"dropped": {}}
        if chat:
            kept = []
            for e in tracker.events:
                if e.get("region") in chat:
                    dropped = report[e["region"]]["dropped"]
                    dropped[e["type"]] = dropped.get(e["type"], 0) + 1
                else:
                    kept.append(e)
            tracker.events = kept
        stats["feed_regions"] = report  # empty: the pass had no feed region (self-checks, #1073)

    def finish(self) -> list[dict[str, Any]]:
        """The events, after the menu frames' sightings are dropped (JSON presets)."""
        tracker, stats = self.interp.tracker, self.stats
        self._drop_chat()
        self.interp.drop_labels()
        self.interp.drop_notice_kills()
        menu: set[float] = set()
        if self.menus:
            feeds = {r.name for r in self.regions if r.kind == "feed"}

            def feed_event(e: dict[str, Any]) -> bool:
                """Feed sightings and kills (kills come from the feed region): never dropped
                but icon kills (:func:`kept`, :func:`feed_activity`); feed rows read as
                notices (``row_notices``) are dropped like other notices.
                This assumes menus do not cover the feed region; a scoreboard drawn over it
                is kept."""
                t = e.get("type")
                return t == "kill" or (e.get("region") in feeds and t != "notice")

            def kept(e: dict[str, Any]) -> bool:
                """Feed events kept on menu frames: not kill lines (:data:`LINE_ENTRIES`), as a
                menu span can hold rows that are no fresh kill, so a line read only inside one
                is dropped even if real. Unless the names have two team colours and the menu
                frames began more than :attr:`MenuParams.static_feed_guard_s` before the line: gameplay
                can pass as a menu, but a coloured menu row opens with its menu."""
                if not feed_event(e):
                    return False
                t0 = e["_times"][0]
                return e.get("entry") not in LINE_ENTRIES or (
                    _two_hues(e)
                    and any(a + MENU.static_feed_guard_s < t0 <= b for a, b in menu_spans)
                )

            # still screens' opening times: menu rows read as icon kills open with the
            # screen, a real kill turns up partway through a still stretch
            opened = [
                run[0]["t"]
                for run in still_stretches(self.frame_boxes)
                if run[0].get("still") is not None and len(run) >= MENU.static_min_frames
            ]

            def feed_activity(e: dict[str, Any]) -> bool:
                """Feed events that guard still frames from the static rule: not icon kills
                first read within MENU.static_feed_guard_s of a still screen's opening."""
                if not feed_event(e):
                    return False
                t0 = e["_times"][0]
                icon = e.get("entry") == "icon"
                return not icon or all(abs(t0 - t) > MENU.static_feed_guard_s for t in opened)

            tracked = tracker.events
            feed_times = {t for e in tracked if feed_activity(e) for t in e["_times"]}
            # new menu text: when texts that are no preset phrase were first read, until when
            # shown; ``end`` moves only on the frames a region is read, so it is padded by one
            # read interval
            step = {r.name: r.every / self.interp.params.fps for r in self.regions}
            unknown = [e for e in tracked if e.get("unknown")]

            def shown(e: dict[str, Any]) -> tuple[float, float]:
                return e["_times"][0], e["end"] + step.get(e.get("region"), 0.0)

            max_gap = MENU.static_max_gap_s if self.still_runs is None else self.still_runs.max_gap
            frames = self.frame_boxes
            lines = [e for e in unknown if e.get("region") not in feeds]
            text_spans = [shown(e) for e in lines]
            text_keys: list[Any] = [(e.get("region"), e["_key"]) for e in lines]  # a line's spans
            for name in sorted(feeds):  # a feed's rows are one stack (see MENU.text_lines)
                rows = [e for e in unknown if e.get("region") == name]
                # the rows that stay as lines of their own, then those on screen together merged
                staying = staying_spans(
                    frames, [shown(e) for e in rows], [(name, e["_key"]) for e in rows], max_gap
                )
                for i, span in enumerate(feed_stacks(staying)):
                    text_spans.append(span)
                    text_keys.append((name, i))
            if self.still_runs is None:
                feed_times, text_spans = None, None
            rules, menu_info = menu_rules(
                frames,
                feed_times=feed_times,
                text_spans=text_spans,
                text_keys=text_keys,
                max_gap=max_gap,
            )
            if menu_logger.isEnabledFor(logging.DEBUG):  # the logger of its lines
                _log_menu_frames(
                    self.src_name,
                    frames,
                    rules,
                    menu_info,
                    tracked,
                    feed_times,
                    text_spans,
                    text_keys,
                    max_gap,
                )
            menu = {f["t"] for f in frames if rules[f["id"]] != "none"}
            menu_spans = merge_menu_frames(menu, self.interp.params.fps)
            tracker.drop_sightings(menu, keep=kept)
            self.interp.digits.drop_sightings(menu)
            stats["menu_frames"] = len(menu)
            stats["menu_spans"] = menu_spans
            stats["static_frames"] = sum(r == "static" for r in rules.values())
            stats["text_frames"] = sum(r == "text" for r in rules.values())
        tracker.events = merge_kill_reads(tracker.events)  # needs the sightings' times
        events = tracker.finish()
        if self.feed_notices:
            feeds = {r.name for r in self.regions if r.kind == "feed"}
            events = [
                e
                for e in events
                if not (
                    e.get("type") == "notice"
                    and e.get("region") in feeds
                    and e["end"] - e["t"] > ROW_NOTICE_MAX_S
                )
            ]
        spans = replay_spans(events)
        if spans:  # kills shown again in a replay are no kills of the player
            step = max(r.every for r in self.regions if r.kind == "replay") / self.interp.params.fps
            stats["replays"] = len(spans)
            stats["replay_kills"] = mark_replay_kills(events, spans, step)
        events = collapse_banner_bursts(events)
        events = learn_announcements(events, self.interp.digits.finish())
        if any(r.kind == "score" for r in self.regions):
            scores, stats["score_slots"] = self.scores.finish(menu)
            events += scores
        for e in events:
            e["kind"] = "ocr"
        return events
