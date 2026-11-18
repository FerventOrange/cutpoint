"""Vision points, OCR reads: the vision HUD map of a game's region discovery (#1069).

Pixel discovery (:mod:`cutpoint.events.discover`) finds text that changes; it can miss a kill feed
that shows rarely and take a loadout box for one. Before the feed labels, the shared vision model
(the one ``cutpoint events`` and ``analyze --vision`` use: Ollama, or a hosted Vision model the
user allowed) is shown up to :data:`MAP_FRAMES` full gameplay frames that discovery already OCR'd,
one call each, and points at the HUD elements that mark strong moments (:data:`WANTED`: kill
feed, score, objective banner, replay marker) and those to ignore (:data:`EXCLUDE`: chat, webcam,
captions), as boxes. It never judges a moment and its words are never stored (``text_seen`` is
logged at DEBUG only).

Vision points, OCR reads:

* **Consensus.** An element counts when it is marked in two frames (same kind, boxes overlapping
  by :data:`MATCH_IOU` IoU), or in one when OCR confirms it strongly (:data:`STRONG_BURSTS`).
  When the first two frames agree (:func:`agree`) the third call is skipped.
* **Validation.** A box becomes a region only when the OCR of discovery's frames (no new OCR)
  finds text in it that fits the kind (:func:`judge`): a kill feed shows feed-like lines
  (:func:`feed_like`), a score small numbers (:func:`~cutpoint.events.discover.score_value`), a
  banner words, a replay marker words that show only some of the time. The region is the union
  of the OCR boxes inside the box (with a margin), padded; at most :data:`MAX_AREA` of the frame
  (a banner: :data:`BANNER_MAX_AREA`, its text in at most :data:`BANNER_MAX_SHARE` of the frames). Chat, webcam and caption boxes seen in two frames are exclusions: no
  rule-made feed there.
* **Merge** (:func:`settle`, :func:`apply`). An element on a pixel proposal (IoU at least
  :data:`MATCH_IOU`) gives it its label (a kill feed: the feed region, as a ``kill_feed`` label
  of the feed labels does); one on no proposal becomes a region of its own, marked
  ``"origin": "vision"``. With a vision kill feed, a rule-made feed elsewhere stays only when its
  OCR reads feed-like lines ("union, OCR judges"). The feed labels then ask only about the
  proposals the map did not settle.

The map is saved in the discovered preset (``discovery.hud_map``), so it runs once per game; an
installed discovered preset without one is topped up on the next analysis (:func:`top_up`). One
call's answer is cached in ``<discovery folder>/hud_map.json``, keyed on the model, the prompt,
the frame, the user's ``--context`` and the footage category.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, progress
from cutpoint.errors import LowMemoryError
from cutpoint.events.discover import (
    REF_VERSION,
    REF_W,
    _alpha_share,
    _icon_lines,
    _lines,
    _norm_text,
    _src,
    boxes_in,
    other_aspect,
    score_region,
    score_value,
)
from cutpoint.events.feed_labels import (
    LABEL_BUDGET,
    STOP_AFTER,
    StopLabelling,
    VisionCalls,
    model_key,
)
from cutpoint.events.regions import REF_H, Region, clamp_box, letters, marker_seen
from cutpoint.imageio import read_image
from cutpoint.utils import context_block, context_hash, file_stamp

logger = logging.getLogger(__name__)

MAP_VERSION = 2  # 2: replay hints; bump when the prompt, schema or frames change: cached answers are not reused
CATEGORY = "gaming"  # the footage the map is for (other categories: #1069 (c))
KINDS = (
    "kill_feed",
    "score",
    "objective_banner",
    "replay",
    "chat",
    "webcam",
    "caption",
    "clock",
    "other",
)
WANTED = ("kill_feed", "score", "objective_banner", "replay")  # become regions
EXCLUDE = ("chat", "webcam", "caption")  # no rule-made feed there
PERSISTENT = ("kill_feed", "score", "chat", "webcam")  # on screen most of the time (agree())
MAP_FRAMES = 3  # frames shown, one call each
MAP_BUDGET = 60  # seconds the map adds to the feed labels' LABEL_BUDGET (one budget per game)
NUM_PREDICT = 900  # reply tokens: about a dozen elements
MATCH_IOU = 0.3  # boxes of one element (frames, proposals, preset regions) overlap this much
MAX_AREA = 0.15  # a region (a banner: BANNER_MAX_AREA) covers at most this share of the frame
BANNER_MAX_AREA = 0.25  # a banner covers at most this share (the largest real one: 0.078)
PAD = 12  # reference px around the OCR boxes, as discover._padded
MARGIN = 0.1  # the vision box grows by this share of its size for the OCR test...
MARGIN_MIN = 16  # ...and at least this many reference px (a model's box is rough)
# kill feed (feed_like): text in FEED_MIN_BURSTS bursts, FEED_MIN_LINES distinct lines, mostly
# letters (names) and either a generic kill word, an icon kill line (two names split by an icon
# gap, discover._icon_lines) or rows that change between the frames of a burst
FEED_MIN_BURSTS = 3
FEED_MIN_LINES = 6
FEED_MIN_ALPHA = 0.45
FEED_MIN_CHANGE = 0.2
SCORE_MIN_BURSTS = 3  # score: a small number (score_value) in this many bursts, 2+ values
BANNER_MIN_BURSTS = 2  # banner: mostly letters, in this many bursts
BANNER_MIN_ALPHA = 0.6
HINT_MAX = 40  # letters kept of what the model read in a replay marker
REPLAY_MIN_BURSTS = 2  # replay marker: words in this many bursts...
REPLAY_MAX_SHARE = 0.5  # ...but in at most this share of the frames (live play is the rule)
BANNER_MAX_SHARE = 0.5  # a banner's text is there in at most this share of the frames
# an element marked in one frame only counts when its OCR test holds in this many bursts
STRONG_BURSTS = {"kill_feed": 6, "score": 6, "objective_banner": 4, "replay": 3}
# the labels a settled proposal gets (feed_labels.LABELS); replay and clock have none
LABEL_OF = {
    "kill_feed": "kill_feed",
    "score": "readout",
    "objective_banner": "banner",
    "chat": "chat",
    "webcam": "other",
    "caption": "other",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "text_seen": {"type": "string"},
                },
                "required": ["kind", "box", "text_seen"],
            },
        }
    },
    "required": ["elements"],
}
# per footage category: the elements wanted; no game's names or words belong here
_WANTED_DEFS = {
    "gaming": (
        "- kill_feed: the game's own list of recent kills or eliminations: rows of a killer's "
        "name, a weapon icon or a verb, and a victim's name.\n"
        "- score: the round or match score: the numbers of points or rounds each side has.\n"
        "- objective_banner: large text announcing a phase, an objective or a result of the "
        "round or match.\n"
        "- replay: a marker that the screen shows a replay or a killcam instead of live play.\n"
    ),
}
_IGNORE_DEFS = (
    "- chat: messages typed by stream viewers or players: a name followed by a message.\n"
    "- webcam: the streamer's camera picture.\n"
    "- caption: subtitles, a stream title or alerts of the streaming software.\n"
    "- clock: a timer or a clock.\n"
    "- other: any other text.\n"
)
_PROMPT = (
    "This is one frame of a recorded game stream. Point out where these elements are on "
    "screen, so that text recognition can read them later:\n"
    "{wanted}"
    "Also mark these, so that they are ignored:\n"
    "{ignore}"
    "{context}"
    "Only point: mark what is visible in this frame and leave out the rest; do not judge the "
    "moment or the play. box: [x0, y0, x1, y1] of the element on a 0-1000 scale of the frame's "
    "width and height. text_seen: a few words of the text in the box, or empty."
)

Rect = tuple[int, int, int, int]  # x, y, w, h in reference px


def prompt(context: str | None = None, category: str = CATEGORY) -> str:
    """The map's prompt for ``category``, with the user's ``--context`` words as a hint."""
    return _PROMPT.format(
        wanted=_WANTED_DEFS[category], ignore=_IGNORE_DEFS, context=context_block(context)
    )


def iou(a: Rect, b: Rect) -> float:
    """Intersection over union of two rectangles ``(x, y, w, h)``."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    inter = max(0, w) * max(0, h)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _union(a: Rect, b: Rect) -> Rect:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    return x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0


def _rect(r: Any) -> Rect:
    """The rectangle of a proposal, a preset region entry or a :class:`Region`."""
    get = r.get if isinstance(r, dict) else lambda k: getattr(r, k)
    return int(get("x")), int(get("y")), int(get("w")), int(get("h"))


def _inside(r: Rect, zone: Rect) -> bool:
    """Whether the centre of ``r`` lies in ``zone``."""
    cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
    return zone[0] <= cx <= zone[0] + zone[2] and zone[1] <= cy <= zone[1] + zone[3]


def _covers(big: Rect, small: Rect) -> bool:
    """Whether at least half of ``small`` lies inside ``big``."""
    w = min(big[0] + big[2], small[0] + small[2]) - max(big[0], small[0])
    h = min(big[1] + big[3], small[1] + small[3]) - max(big[1], small[1])
    return w > 0 and h > 0 and w * h >= 0.5 * small[2] * small[3]


def _numbers(box: Any) -> list[float] | None:
    """The four finite numbers of a model box, or None."""
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    return vals if all(np.isfinite(vals)) else None


def to_ref(box: Any, size: tuple[int, int], pixels: bool | None = None) -> Rect | None:
    """A model box ``[x0, y0, x1, y1]`` in reference px, clamped to the frame (None when it is
    no box or nothing of it is inside). Values on a 0-1000 scale of the image (Qwen-VL's
    grounding) unless one is over 1000: then pixels of the image sent (``size``: width,
    height); ``pixels`` decides it instead (:func:`parse`: once per reply). Discovery's frames keep the source's aspect; x and y scale on their
    own (:func:`cutpoint.events.discover._ocr_frames`), so the boxes land where OCR's do."""
    vals = _numbers(box)
    if vals is None:
        return None
    w, h = size
    if pixels is None:
        pixels = max(vals) > 1000
    if pixels:
        px = vals
    else:
        px = [vals[0] / 1000 * w, vals[1] / 1000 * h, vals[2] / 1000 * w, vals[3] / 1000 * h]
    x0, x1 = sorted((px[0], px[2]))
    y0, y1 = sorted((px[1], px[3]))
    sx, sy = REF_W / w, REF_H / h
    return clamp_box(round(x0 * sx), round(y0 * sy), round((x1 - x0) * sx), round((y1 - y0) * sy))


def parse(reply: Any, size: tuple[int, int]) -> list[dict[str, Any]] | None:
    """The elements of a model ``reply`` (``{"kind", "box"}``, the box in reference px); None
    when the reply has no element list. Elements of unknown kinds or without a usable box are
    left out; what the model read is logged at DEBUG only, never kept."""
    els = reply.get("elements") if isinstance(reply, dict) else None
    if not isinstance(els, list):
        return None
    # one scale for the whole reply: pixels when any box has a value over 1000
    nums = [v for e in els if isinstance(e, dict) for v in (_numbers(e.get("box")) or ())]
    pixels = bool(nums) and max(nums) > 1000
    out = []
    for e in els:
        if not isinstance(e, dict) or e.get("kind") not in KINDS:
            continue
        box = to_ref(e.get("box"), size, pixels)
        if box is None or box[2] * box[3] < 16:
            continue
        logger.debug("HUD map element %s at %s (read %.60r)", e["kind"], box, e.get("text_seen"))
        el: dict[str, Any] = {"kind": e["kind"], "box": list(box)}
        if e["kind"] == "replay" and isinstance(e.get("text_seen"), str):
            # which of the texts OCR reads there is the marker (:func:`replay_marker`); kept to
            # its letters only, never logged above DEBUG nor written into the preset
            el["hint"] = letters(e["text_seen"])[:HINT_MAX]
        out.append(el)
    return out


def _stored(els: Any) -> list[dict[str, Any]] | None:
    """Cached elements, checked (a hand-edited or older file never crashes)."""
    if not isinstance(els, list):
        return None
    out = []
    for e in els:
        if isinstance(e, dict) and e.get("kind") in KINDS:
            box = e.get("box")
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box):
                got = clamp_box(*box)
                if got is not None:
                    el: dict[str, Any] = {"kind": e["kind"], "box": list(got)}
                    if e["kind"] == "replay" and isinstance(e.get("hint"), str):
                        el["hint"] = letters(e["hint"])[:HINT_MAX]
                    out.append(el)
    return out


def pick_frames(
    proposals: list[dict[str, Any]], frames: list[dict[str, Any]], frame_dir: Path, n: int
) -> list[dict[str, Any]]:
    """Up to ``n`` gameplay frames to show, one per burst: each the frame that shows text in the
    most proposals not covered by the frames before (readouts, popups and score slots left
    out), then the one with the most proposals and boxes; so a feed, banners and a replay marker
    that show at times are likelier to be on one of them."""
    props = [p for p in proposals if p["kind"] not in ("readout", "popup", "score")]
    pool = [f for f in frames if (frame_dir / str(f.get("frame"))).is_file()]
    cover = {
        f["id"]: {p["name"] for p in props if boxes_in(f, p["x"], p["y"], p["w"], p["h"])}
        for f in pool
    }
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    bursts: set[tuple[str, Any]] = set()
    for _ in range(n):
        left = [f for f in pool if (_src(f), f.get("burst")) not in bursts]
        if not left:
            break
        best = max(
            left,
            key=lambda f: (
                len(cover[f["id"]] - seen),
                len(cover[f["id"]]),
                len(f["boxes"]),
                str(f["id"]),
            ),
        )
        chosen.append(best)
        seen |= cover[best["id"]]
        bursts.add((_src(best), best.get("burst")))
    return chosen


def group(seen: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The elements of the frames ``seen`` grouped: one kind, boxes overlapping by
    :data:`MATCH_IOU`, one per frame; ``{"kind", "box" (their union), "frames"}``."""
    groups: list[dict[str, Any]] = []
    for i, els in enumerate(seen):
        for e in els:
            box = tuple(e["box"])
            g = next(
                (
                    g
                    for g in groups
                    if g["kind"] == e["kind"]
                    and i not in g["frames"]
                    and iou(g["box"], box) >= MATCH_IOU
                ),
                None,
            )
            if g is None:
                g = {"kind": e["kind"], "box": box, "frames": {i}, "hints": []}
                groups.append(g)
            else:
                g["box"] = _union(g["box"], box)
                g["frames"].add(i)
            if e.get("hint"):
                g["hints"].append(e["hint"])
    return groups


def agree(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """Whether two frames' elements agree, so a third frame is not asked: at least one element
    marked in both, and every one of the kinds on screen most of the time (:data:`PERSISTENT`)
    marked in both."""
    groups = group([a, b])
    return any(len(g["frames"]) == 2 for g in groups) and all(
        len(g["frames"]) == 2 for g in groups if g["kind"] in PERSISTENT
    )


def _grow(r: Rect) -> Rect:
    mx, my = max(MARGIN_MIN, MARGIN * r[2]), max(MARGIN_MIN, MARGIN * r[3])
    got = clamp_box(round(r[0] - mx), round(r[1] - my), round(r[2] + 2 * mx), round(r[3] + 2 * my))
    return got or r


def ocr_stats(frames: list[dict[str, Any]], zone: Rect) -> dict[str, Any]:
    """What discovery's OCR read in ``zone`` over the gameplay ``frames`` (counts only, no
    text): frames and bursts with text, distinct lines, the change rate (consecutive frames of a
    burst whose text there differs, of those with text in either), generic kill words, icon
    kill lines, the letter share and the small numbers (:func:`score_value`)."""
    from cutpoint.events.killfeed import DEFAULT_KILL_WORDS  # noqa: PLC0415 (heavy imports)

    x, y, w, h = zone
    per = [(f, boxes_in(f, x, y, w, h)) for f in frames]
    inside = [(f, bs) for f, bs in per if bs]
    lines = [" ".join(b["text"] for b in ln) for _, bs in inside for ln in _lines(bs)]
    by_burst: dict[tuple[str, Any], list[tuple[int, str]]] = {}
    for f, bs in per:
        text = "".join(sorted(_norm_text(b["text"]) for b in bs))
        by_burst.setdefault((_src(f), f.get("burst")), []).append((int(f.get("k", 0)), text))
    pairs = changed = 0
    for fl in by_burst.values():
        fl.sort()
        for (_, a), (_, b) in pairwise(fl):
            if a or b:
                pairs += 1
                changed += a != b
    words = [w for w in DEFAULT_KILL_WORDS if " " not in w]
    kill_lines = sum(any(w in t.lower() for w in words) for t in lines)
    _, icon_bursts, _, _ = _icon_lines(inside)
    digits = [(f, b) for f, bs in inside for b in bs if score_value(b["text"]) is not None]
    boxes = [b for _, bs in inside for b in bs]
    return {
        "frames": len(inside),
        "share": round(len(inside) / max(1, len(frames)), 3),
        "bursts": len({(_src(f), f.get("burst")) for f, _ in inside}),
        "lines": len({_norm_text(t) for t in lines} - {""}),
        "change": round(changed / max(1, pairs), 3),
        "kill_lines": kill_lines,
        "icon_bursts": icon_bursts,
        "alpha": round(_alpha_share([b["text"] for b in boxes]), 3),
        "digit_bursts": len({(_src(f), f.get("burst")) for f, _ in digits}),
        "values": sorted({v for _, b in digits if (v := score_value(b["text"])) is not None}),
        "_boxes": boxes,
        "_digits": [b for _, b in digits],
    }


def feed_like(frames: list[dict[str, Any]], rect: Rect) -> tuple[bool, dict[str, Any]]:
    """Whether OCR reads feed-like lines in ``rect`` over the gameplay ``frames``: text in
    :data:`FEED_MIN_BURSTS` bursts, :data:`FEED_MIN_LINES` distinct lines, mostly letters, and a
    generic kill word, an icon kill line or rows that change (:data:`FEED_MIN_CHANGE`). A loadout
    box or a label (few lines, no kill words, no changing rows) is not. Returns the verdict and
    the counts (:func:`ocr_stats`, without the boxes)."""
    st = ocr_stats(frames, rect)
    ok = (
        st["bursts"] >= FEED_MIN_BURSTS
        and st["lines"] >= FEED_MIN_LINES
        and st["alpha"] >= FEED_MIN_ALPHA
        and (st["kill_lines"] > 0 or st["icon_bursts"] > 0 or st["change"] >= FEED_MIN_CHANGE)
    )
    return ok, {k: v for k, v in st.items() if not k.startswith("_")}


def _max_area(kind: str) -> float:
    """The largest share of the frame a region of ``kind`` covers."""
    return BANNER_MAX_AREA if kind == "objective_banner" else MAX_AREA


def _test(kind: str, st: dict[str, Any]) -> str | None:
    """Why OCR does not confirm an element of ``kind`` (None: it does)."""
    if kind == "kill_feed":
        if st["bursts"] < FEED_MIN_BURSTS:
            return "OCR text in too few bursts"
        if st["lines"] < FEED_MIN_LINES or st["alpha"] < FEED_MIN_ALPHA:
            return "OCR reads too few name lines"
        if not (st["kill_lines"] or st["icon_bursts"] or st["change"] >= FEED_MIN_CHANGE):
            return "OCR reads no kill lines and no changing rows"
        return None
    if kind == "score":
        if st["digit_bursts"] < SCORE_MIN_BURSTS or len(st["values"]) < 2:
            return "OCR reads no score digits that change"
        return None
    if kind == "objective_banner":
        if st["bursts"] < BANNER_MIN_BURSTS or st["alpha"] < BANNER_MIN_ALPHA:
            return "OCR reads no banner words"
        if st["share"] > BANNER_MAX_SHARE:
            return "text there most of the time, no banner"
        return None
    if kind == "replay":
        if st["bursts"] < REPLAY_MIN_BURSTS or st["alpha"] < BANNER_MIN_ALPHA:
            return "OCR reads no marker words"
        if st["share"] > REPLAY_MAX_SHARE:
            return "text there most of the time, no replay marker"
        return None
    return "not a wanted kind"


def _off_feeds(frames: list[dict[str, Any]], feeds: list[Rect]) -> list[dict[str, Any]]:
    """``frames`` without the OCR boxes whose centre lies in one of the ``feeds``."""
    if not feeds:
        return frames
    out = []
    for f in frames:
        keep = [
            b
            for b in f["boxes"]
            if not any(
                x <= (b["x0"] + b["x1"]) / 2 <= x + w and y <= (b["y0"] + b["y1"]) / 2 <= y + h
                for x, y, w, h in feeds
            )
        ]
        out.append(f | {"boxes": keep})
    return out


def _clip_out(rect: Rect, other: Rect) -> Rect | None:
    """The largest part of ``rect`` beside ``other`` (above, below, left or right of it);
    ``rect`` itself when they do not overlap, None when nothing is left."""
    x, y, w, h = rect
    ox, oy, ow, oh = other
    if ox >= x + w or ox + ow <= x or oy >= y + h or oy + oh <= y:
        return rect
    parts = [
        (x, y, w, oy - y),
        (x, oy + oh, w, y + h - oy - oh),
        (x, y, ox - x, h),
        (ox + ow, y, x + w - ox - ow, h),
    ]
    best = max(parts, key=lambda r: r[2] * r[3] if r[2] > 0 and r[3] > 0 else 0)
    return best if best[2] > 0 and best[3] > 0 else None


def replay_marker(
    frames: list[dict[str, Any]],
    zone: Rect,
    hints: list[str] | tuple[str, ...] = (),
    feeds: list[Rect] | tuple[Rect, ...] = (),
) -> str:
    """The replay marker's text in ``zone``: of the OCR lines there (none whose box centre
    lies in a kill feed, ``feeds``: the streamer's own kill line is no marker), grouped by
    :func:`cutpoint.events.regions.marker_seen` (a misread "DREPLAY" joins "REPLAY"), the group
    read in at least :data:`REPLAY_MIN_BURSTS` bursts that is like what the model read in the
    marker (``hints``), as its most read line; "" when no such group is (no hint, or no OCR
    text like it). OCR reads the text from the footage, so no game's words are built in; the
    model's words only pick among those texts (a map name shown in the same place at a round's
    start is read as often as the marker)."""
    seen: dict[str, set[tuple[str, Any]]] = {}
    for f in _off_feeds(frames, list(feeds)):
        for ln in _lines(boxes_in(f, *zone)):
            text = letters(" ".join(b["text"] for b in ln))
            if len(text) >= 3:
                seen.setdefault(text, set()).add((_src(f), f.get("burst")))
    best, best_n, hinted = "", 0, False
    for text in sorted(seen, key=lambda t: (-len(seen[t]), t)):
        group = set().union(*(b for t, b in seen.items() if marker_seen(t, text)))
        if len(group) < REPLAY_MIN_BURSTS:
            continue
        like = any(marker_seen(h, text) for h in hints)
        if like and (not hinted or len(group) > best_n):
            best, best_n, hinted = text, len(group), like
    return best


def judge(
    g: dict[str, Any],
    frames: list[dict[str, Any]],
    answered: int,
    feeds: list[Rect] | tuple[Rect, ...] = (),
) -> dict[str, Any]:
    """The record of an element group ``g`` (:func:`group`): ``confirmed`` and its region
    rectangle ``rect`` (reference px) when OCR confirms it (:func:`_test`), else ``why`` not.
    ``answered``: frames the model answered for. A replay marker is judged without the OCR
    boxes in the kill ``feeds`` and its rectangle leaves them out."""
    kind, n = g["kind"], len(g["frames"])
    if kind == "replay":
        frames = _off_feeds(frames, list(feeds))
    rec: dict[str, Any] = {"kind": kind, "box": list(g["box"]), "seen": n, "confirmed": False}
    if kind in EXCLUDE:
        rec["confirmed"] = n >= 2
        if n < 2:
            rec["why"] = "marked in one frame"
        return rec
    if kind not in WANTED:
        rec["why"] = "ignored kind"
        return rec
    zone = _grow(g["box"])
    st = ocr_stats(frames, zone)
    rec["evidence"] = {k: v for k, v in st.items() if not k.startswith("_")}
    why = _test(kind, st)
    if why is None and n < 2 and st["bursts"] < STRONG_BURSTS[kind]:
        why = "marked in one frame, OCR too weak"
    if why is not None:
        rec["why"] = why
        return rec
    bs = st["_digits"] if kind == "score" else st["_boxes"]
    x0 = max(zone[0], min(b["x0"] for b in bs))
    y0 = max(zone[1], min(b["y0"] for b in bs))
    x1 = min(zone[0] + zone[2], max(b["x1"] for b in bs))
    y1 = min(zone[1] + zone[3], max(b["y1"] for b in bs))
    if kind == "score":  # score_region pads it and leaves room for a second digit
        rect = clamp_box(x0, y0, x1 - x0, y1 - y0)
    else:
        rect = clamp_box(x0 - PAD, y0 - PAD, x1 - x0 + 2 * PAD, y1 - y0 + 2 * PAD)
    if rect is None:
        rec["why"] = "no OCR box inside"
        return rec
    if rect[2] * rect[3] > _max_area(kind) * REF_W * REF_H:
        rec["why"] = "too large"
        return rec
    if kind == "score":
        rec["median_text_h"] = round(float(np.median([b["y1"] - b["y0"] for b in bs])), 1)
    if kind == "replay":
        marker = replay_marker(frames, zone, g.get("hints") or ())
        if not marker:
            rec["why"] = "the model read no marker text OCR reads again and again"
            return rec
        for f in feeds:
            got = _clip_out(rect, f)
            if got is None:
                rec["why"] = "inside a kill feed"
                return rec
            rect = got
        rec["marker"] = marker
    rec |= {"confirmed": True, "rect": list(rect)}
    return rec


def _key(model: str, frame_id: str, context: str | None, category: str) -> str:
    return cache.params_hash(
        {
            "v": MAP_VERSION,
            "model": model,
            "prompt": cache.params_hash(
                {"p": _PROMPT, "s": SCHEMA, "w": _WANTED_DEFS, "i": _IGNORE_DEFS}, 8
            ),
            "frame": frame_id,
            "context": context_hash(context),
            "category": category,
        },
        16,
    )


def map_hud(
    proposals: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    frame_dir: Path,
    ask: Callable[..., dict[str, Any] | None],
    model: str,
    context: str | None = None,
    category: str = CATEGORY,
) -> dict[str, Any] | None:
    """Ask the vision model where the HUD elements are on up to :data:`MAP_FRAMES` frames and
    keep what OCR confirms.

    Args:
        proposals: Discovery's proposals (to pick frames that show them).
        frames: The gameplay frames discovery OCR'd.
        frame_dir: Folder of their JPEGs (``frames/``); the cache file is next to it.
        ask: ``(prompt, jpegs, schema, num_predict) -> reply dict or None``; may raise
            :class:`~cutpoint.events.feed_labels.StopLabelling`.
        model: The model's cache key (provider, host and model id).
        context: The user's ``--context`` words, a hint in the prompt.
        category: The footage category (:data:`CATEGORY`).

    Returns:
        ``{"v", "model", "category", "frames" (ids answered), "calls", "elements"}`` (each
        element as :func:`judge` makes it), or None when the model answered for no frame.
    """
    picked = pick_frames(proposals, frames, frame_dir, MAP_FRAMES)
    if not picked:
        return None
    path = frame_dir.parent / "hud_map.json"
    loaded = cache.load_json(path)
    store: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    text = prompt(context, category)
    seen: list[list[dict[str, Any]]] = []
    used: list[str] = []
    calls = failed = 0
    for f in picked:
        if len(seen) == 2 and agree(seen[0], seen[1]):
            logger.debug("HUD map: the first two frames agree; the third is not asked")
            break
        key = _key(model, str(f["id"]), context, category)
        hit = store.get(key)
        els = _stored(hit.get("elements")) if isinstance(hit, dict) else None
        if (
            els is not None
            and hit.get("ref_v") != REF_VERSION
            and other_aspect(frame_dir / f["frame"])
        ):
            els = None  # boxes made before y scaled with the height (#1085): ask again
        if els is None:
            img = read_image(frame_dir / f["frame"])
            if img is None:
                continue
            try:
                reply = ask(text, [(frame_dir / f["frame"]).read_bytes()], SCHEMA, NUM_PREDICT)
            except StopLabelling as why:
                logger.info("HUD map: %s; going on with the frames answered", why)
                break
            calls += 1
            els = parse(reply, (img.shape[1], img.shape[0]))
            if els is None:
                failed += 1
                if failed >= STOP_AFTER:
                    logger.info("HUD map: the vision model gave no usable answer; stopping")
                    break
                continue
            failed = 0
            store[key] = {"elements": els, "ref_v": REF_VERSION}
            cache.save_json(path, store)
        seen.append(els)
        used.append(str(f["id"]))
    if not seen:
        return None
    groups = group(seen)
    # the replay marker last, without the kill feeds' text (a feed row is no marker)
    recs = {i: judge(g, frames, len(seen)) for i, g in enumerate(groups) if g["kind"] != "replay"}
    feeds = [tuple(r["rect"]) for r in recs.values() if r["confirmed"] and r["kind"] == "kill_feed"]
    feeds += [_rect(p) for p in proposals if p.get("kind") == "feed"]
    for i, g in enumerate(groups):
        if g["kind"] == "replay":
            recs[i] = judge(g, frames, len(seen), feeds)
    elements = [recs[i] for i in range(len(groups))]
    return {
        "v": MAP_VERSION,
        "model": model,
        "category": category,
        "frames": used,
        "calls": calls,
        "elements": elements,
    }


def _name(base: str, taken: set[str]) -> str:
    k, name = 1, base
    while name in taken:
        k += 1
        name = f"{base}{k}"
    taken.add(name)
    return name


def new_region(e: dict[str, Any], name: str) -> Region:
    """The region of a confirmed element ``e`` (:func:`judge`) of a wanted kind: a ``feed``
    on the det gate at the full rate, a ``score`` readout (:func:`score_region`), a ``banner``
    every 3rd frame or a ``replay`` marker on every frame."""
    x, y, w, h = e["rect"]
    if e["kind"] == "score":
        p = {
            "name": name,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "evidence": {
                "values": e.get("evidence", {}).get("values") or [10],
                "median_text_h": e.get("median_text_h", h),
            },
        }
        return score_region(p)
    kind, every = {"kill_feed": ("feed", 1), "objective_banner": ("banner", 3)}.get(
        e["kind"], ("replay", 1)
    )
    return Region(name, x, y, w, h, kind, gate="det", every=every, marker=e.get("marker", ""))


_BASE_NAMES = {
    "kill_feed": "vision_feed",
    "score": "vision_score",
    "objective_banner": "vision_banner",
    "replay": "replay",
}


def _on_banner(rect: Rect, elements: list[dict[str, Any]], regions: list[dict[str, Any]]) -> bool:
    """Whether a replay marker's ``rect`` lies on a banner (a confirmed objective banner of the
    map, or a banner or notice region or proposal, IoU at least :data:`MATCH_IOU`): banner text
    such as a bomb notice is no replay marker (INFO)."""
    boxes = [
        tuple(e["rect"])
        for e in elements
        if e.get("confirmed") and e["kind"] == "objective_banner" and e.get("rect")
    ]
    for r in regions:
        if r.get("kind") in ("banner", "notice"):
            try:
                boxes.append(_rect(r))
            except (TypeError, ValueError, KeyError):
                continue
    if any(iou(rect, b) >= MATCH_IOU for b in boxes):
        logger.info("HUD map: the replay marker at %d,%d %dx%d lies on a banner; left out", *rect)
        return True
    return False


def settle(
    hud: dict[str, Any], proposals: list[dict[str, Any]], taken: set[str]
) -> tuple[dict[str, dict[str, str]], list[Region], list[Rect]]:
    """What the map's confirmed elements make of discovery's proposals.

    Returns:
        (labels, regions, exclusions): labels of the proposals an element lies on (IoU at least
        :data:`MATCH_IOU`, or, for a score readout, every score proposal at least half inside it; as :func:`~cutpoint.events.discover.apply_labels` takes them, with
        ``"from": "hud_map"``), new regions (``"origin": "vision"`` in the preset) for elements
        on no proposal, and the chat, webcam and caption boxes. ``taken``: names in use (the
        new regions' names are added).
    """
    labels: dict[str, dict[str, str]] = {}
    regions: list[Region] = []
    excl: list[Rect] = []
    model = str(hud.get("model", ""))
    fits = {
        "kill_feed": lambda p: p["kind"] != "score",
        "score": lambda p: p["kind"] == "score",
        "objective_banner": lambda p: p["kind"] in ("banner", "notice"),
        "replay": lambda _p: False,  # read in a way of its own: always a region
    }
    # the wanted kinds first: a proposal a kill feed lies on keeps that label
    confirmed = [e for e in hud.get("elements") or [] if e.get("confirmed")]
    for e in sorted(confirmed, key=lambda e: e["kind"] in EXCLUDE):
        kind = e["kind"]
        if kind in EXCLUDE:
            excl.append(tuple(e["box"]))
            for p in proposals:
                if p["name"] not in labels and (
                    iou(_rect(p), tuple(e["box"])) >= MATCH_IOU
                    or _inside(_rect(p), tuple(e["box"]))
                ):
                    labels[p["name"]] = {"label": LABEL_OF[kind], "model": model, "from": "hud_map"}
            continue
        rect = tuple(e["rect"])
        if rect[2] * rect[3] > _max_area(kind) * REF_W * REF_H:
            continue
        if kind == "replay" and _on_banner(rect, confirmed, proposals):
            continue
        on = [(iou(_rect(p), rect), p) for p in proposals if fits[kind](p)]
        if kind == "score":  # score proposals inside a wide readout are it: all labelled
            inner = [p for p in proposals if fits[kind](p) and _covers(rect, _rect(p))]
            for p in inner:
                labels[p["name"]] = {"label": LABEL_OF[kind], "model": model, "from": "hud_map"}
            if inner:
                continue
        best = max(on, key=lambda t: t[0], default=None)
        if best is not None and best[0] >= MATCH_IOU:
            labels[best[1]["name"]] = {"label": LABEL_OF[kind], "model": model, "from": "hud_map"}
            continue
        regions.append(new_region(e, _name(_BASE_NAMES[kind], taken)))
    return labels, regions, excl


def apply(
    regions: list[Region],
    extra: list[dict[str, Any]],
    new: list[Region],
    excl: list[Rect],
    labels: dict[str, dict[str, str]],
    frames: list[dict[str, Any]],
) -> tuple[list[Region], list[dict[str, Any]]]:
    """Discovery's regions after the map (:func:`settle`): a rule-made feed in a chat box stays
    a ``feed`` (its lines are filtered when it is read); with a kill feed the map confirmed, a
    rule-made feed elsewhere that no model named a kill feed stays a feed only when OCR reads
    feed-like lines in it (:func:`feed_like`; else a ``notice``); the new regions are added
    (feeds first) and an outside notice region on a new feed leaves ``extra``."""
    named = {n for n, v in labels.items() if v.get("label") == "kill_feed"}
    vision_feeds = [r for r in new if r.kind == "feed"]
    judged = bool(frames) and (
        bool(vision_feeds)
        or any(
            v.get("from") == "hud_map" and v.get("label") == "kill_feed" for v in labels.values()
        )
    )
    out: list[Region] = []
    for r in regions:
        if r.kind == "feed" and r.name not in named:
            excluded = any(_inside(_rect(r), x) for x in excl)
            bogus = (
                judged
                and r.name not in labels
                and not any(iou(_rect(r), _rect(v)) >= MATCH_IOU for v in vision_feeds)
                and not feed_like(frames, _rect(r))[0]
            )
            if excluded:
                # stays a feed: reading it tells chat from a feed by lines per hour and drops
                # the events of a chat (killfeed._drop_chat), as with a feed labelled chat
                logger.info(
                    "HUD map: rule-made feed %s@%d,%d %dx%d is no kill feed (a chat "
                    "box there); kept as a feed so its lines are filtered",
                    r.name,
                    *_rect(r),
                )
            elif bogus:
                logger.info(
                    "HUD map: rule-made feed %s@%d,%d %dx%d is no kill feed (OCR reads no feed "
                    "lines); read as a notice",
                    r.name,
                    *_rect(r),
                )
                r = Region(r.name, r.x, r.y, r.w, r.h, "notice", gate=r.gate, every=3)
        out.append(r)
    feeds = [r for r in new if r.kind == "feed"]
    others = [r for r in new if r.kind != "feed"]
    extra = [e for e in extra if not any(iou(_rect(e), _rect(v)) >= MATCH_IOU for v in feeds)]
    return feeds + out + others, extra


def top_up(
    data: dict[str, Any], hud: dict[str, Any], frames: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """An installed discovered preset ``data`` with the confirmed elements of the map ``hud``
    added, run once on a project's frames (``frames``: its gameplay frames, OCR'd).

    * A kill feed on one of the preset's feed regions (IoU at least :data:`MATCH_IOU`) only labels
      it. Else union, OCR judges: each feed region stays a feed only when OCR reads feed-like
      lines in it (:func:`feed_like`; a loadout box becomes a ``notice``, ``"was": "feed"``),
      and the feed is added, taking over a notice or banner region it lies on (hand edits
      included: the rewritten entries are kept as they were in ``top_up.originals``); the preset gets ``"feed": {"row_notices": true}``
      unless its ``feed`` lists ``kill_words``.
    * A score readout, an objective banner or a replay marker is added where the preset has no
      region (and no score or replay region of its kind yet).
    * ``discovery.hud_map`` records the map and what changed, so it runs once per game.

    Returns:
        (the new preset dict, ``{"added", "replaced", "relabelled"}`` region names).
    """
    data = dict(data)
    regs = [dict(r) for r in data.get("regions") or [] if isinstance(r, dict)]
    taken = {str(r.get("name")) for r in regs}
    changes: dict[str, list[str]] = {"added": [], "replaced": [], "relabelled": []}
    originals: list[dict[str, Any]] = []  # the entries a change rewrote, as they were

    def rect_of(r: dict[str, Any]) -> Rect | None:
        try:
            return _rect(r)
        except (TypeError, ValueError, KeyError):
            return None

    def overlaps(rect: Rect, rs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        got = [(iou(q, rect), r) for r in rs if (q := rect_of(r)) is not None]
        return [r for v, r in sorted(got, key=lambda t: -t[0]) if v >= MATCH_IOU]

    feed_added = False
    # the replay marker last, so it leaves out the feed a vision kill feed added
    for e in sorted(hud.get("elements") or [], key=lambda e: e.get("kind") == "replay"):
        if not e.get("confirmed") or e["kind"] not in WANTED:
            continue
        rect = tuple(e["rect"])
        kind = e["kind"]
        if kind == "kill_feed":
            same = overlaps(rect, [r for r in regs if r.get("kind") == "feed"])
            if same:
                for r in same:
                    if r.get("label") != "kill_feed":
                        r["label"] = "kill_feed"
                        changes["relabelled"].append(str(r.get("name")))
                continue
            for i, r in enumerate(regs):
                q = rect_of(r) if frames and r.get("kind") == "feed" else None
                if q is not None and not feed_like(frames, q)[0]:
                    # read as a notice, as discovery does with a rule feed (:func:`apply`); the
                    # entry as it was is kept in ``discovery.hud_map.top_up``
                    originals.append(dict(r))
                    regs[i] = r | {"kind": "notice", "every": 3, "was": "feed"}
                    changes["replaced"].append(str(r.get("name")))
            over = overlaps(
                rect, [r for r in regs if r.get("kind") in ("notice", "banner", "dynamic")]
            )
            feed = new_region(e, over[0]["name"] if over else _name("vision_feed", taken)).key()
            entry = feed | {"origin": "vision", "label": "kill_feed"}
            if over:
                originals.append(dict(over[0]))
                regs[regs.index(over[0])] = entry
                changes["relabelled"].append(str(entry["name"]))
            else:
                regs.insert(0, entry)
                changes["added"].append(str(entry["name"]))
            feed_added = True
            continue
        if kind == "replay":
            if _on_banner(rect, hud.get("elements") or [], regs):
                continue
            for q in [rect_of(r) for r in regs if r.get("kind") == "feed"]:
                if q is not None and rect is not None:
                    rect = _clip_out(rect, q)
            if rect is None:
                continue
            e = e | {"rect": list(rect)}
        if rect[2] * rect[3] > _max_area(kind) * REF_W * REF_H:
            continue
        made = new_region(e, "_").kind
        if overlaps(rect, regs) or (
            made in ("score", "replay") and any(r.get("kind") == made for r in regs)
        ):
            continue
        entry = new_region(e, _name(_BASE_NAMES[kind], taken)).key() | {"origin": "vision"}
        regs.append(entry)
        changes["added"].append(str(entry["name"]))
    data["regions"] = regs
    feed_cfg = dict(data.get("feed") or {}) if isinstance(data.get("feed"), dict) else {}
    if feed_added and not feed_cfg.get("kill_words"):
        data["feed"] = feed_cfg | {"row_notices": True}
    disc = dict(data.get("discovery") or {}) if isinstance(data.get("discovery"), dict) else {}
    data["discovery"] = disc | {"hud_map": hud | {"top_up": changes | {"originals": originals}}}
    return data, changes


def wants_top_up(preset: Any, presets_dir: Path) -> bool:
    """Whether an existing preset gets the map once (:func:`top_up`): a discovered preset
    installed in the user's presets folder ``presets_dir`` (never a project one)
    whose ``discovery`` has no ``hud_map`` yet."""
    path = getattr(preset, "path", None)
    if path is None or getattr(preset, "source", "") != "discovered":
        return False
    try:
        if Path(path).resolve().parent != Path(presets_dir).resolve():
            return False
    except OSError:
        return False
    data = cache.load_json(path)
    disc = data.get("discovery") if isinstance(data, dict) else None
    return isinstance(disc, dict) and "hud_map" not in disc


TRIED_FILE = "hud_map_tried.json"  # in the cache folder: top-ups that gave no map


def _tried_key(game: str, model: str) -> str:
    return f"{game}|{model}"


def _stamp(path: Any) -> list[int] | None:
    stamp = file_stamp(Path(path))
    return None if stamp is None else [stamp[1], stamp[0]]


def tried(game: str, model: str, preset_path: Any) -> bool:
    """Whether a top-up of ``game``'s preset with ``model`` gave no map before
    (:func:`mark_tried`) and the preset file is unchanged since: not tried again then."""
    from cutpoint import paths as cutpoint_paths  # noqa: PLC0415

    data = cache.load_json(cutpoint_paths.cache_dir() / TRIED_FILE)
    rec = data.get(_tried_key(game, model)) if isinstance(data, dict) else None
    stamp = _stamp(preset_path)
    return isinstance(rec, dict) and stamp is not None and rec.get("preset") == stamp


def mark_tried(game: str, model: str, preset_path: Any, why: str) -> None:
    """Record that a top-up of ``game``'s preset with ``model`` gave no map (``why``), outside
    the preset, so the multi-minute discovery is not run again for it while the preset file
    stays as it is (an edited preset or another model tries again). Not written: DEBUG."""
    from cutpoint import paths as cutpoint_paths  # noqa: PLC0415

    path = cutpoint_paths.cache_dir() / TRIED_FILE
    data = cache.load_json(path)
    data = dict(data) if isinstance(data, dict) else {}
    data[_tried_key(game, model)] = {"preset": _stamp(preset_path), "why": why}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.save_json(path, data)
    except OSError:
        logger.debug("Could not record the HUD map attempt in %s", path, exc_info=True)


def describe(hud: dict[str, Any]) -> str:
    """The map's elements in one line (kinds, rectangles and verdicts; no text)."""
    parts = []
    for e in hud.get("elements") or []:
        where = "{}@{},{} {}x{}".format(e["kind"], *(e.get("rect") or e["box"]))
        verdict = (
            ("excluded" if e["kind"] in EXCLUDE else "confirmed")
            if e.get("confirmed")
            else f"not kept: {e.get('why', '?')}"
        )
        parts.append(f"{where} ({e['seen']} frame(s), {verdict})")
    return ", ".join(parts) or "nothing marked"


class HudMapper:
    """The vision HUD map for ``discover_regions`` (:func:`make_mapper`): called with the
    proposals, the gameplay frames and their folder, it returns the map (:func:`map_hud`) or
    None. It shares ``calls`` (the model check, load and budget) with the feed labels."""

    def __init__(self, rater: Any, context: str | None, calls: VisionCalls) -> None:
        self.rater = rater
        self.context = context
        self.calls = calls
        self._said = False

    def usable(self, *, top_up: bool = False) -> bool:
        """Whether the model answers. For a top-up of an existing preset (``top_up``) a
        provider without the Vision consent only gives an INFO line (no warning: nothing
        else of the run asked for it)."""
        if top_up and self.rater.consent_needed:
            if not self._said:
                self._said = True
                logger.info(
                    "HUD map: the vision model needs your Vision consent; existing HUD presets "
                    "are not topped up"
                )
            return False
        return self.calls.usable()

    @property
    def model(self) -> str:
        """The model's cache key (:func:`cutpoint.events.feed_labels.model_key`)."""
        return model_key(self.rater)

    def ready(self) -> str | None:
        """None when calls can go out now (an Ollama model loaded on a GPU,
        :meth:`VisionCalls.ready`), else why not."""
        if self.calls.ready():
            return None
        return self.calls.stop[0] if self.calls.stop else "no vision model answers"

    def __call__(
        self, proposals: list[dict[str, Any]], frames: list[dict[str, Any]], frame_dir: Path
    ) -> dict[str, Any] | None:
        if not frames or not self.calls.usable():
            return None
        self.calls.begin(frame_dir, LABEL_BUDGET + MAP_BUDGET)
        progress.update(detail="mapping the HUD with the vision model")
        try:
            hud = map_hud(
                proposals, frames, frame_dir, self.calls.ask, model_key(self.rater), self.context
            )
        except LowMemoryError as err:
            logger.warning("HUD map skipped, the rules decide: %s", err)
            self.calls.release()
            return None
        self.calls.carry_over(frame_dir)  # the feed labels go on with this budget
        if hud is not None:
            logger.info("HUD map (%s): %s", hud["model"], describe(hud))
        return hud

    def release(self) -> None:
        """Free an Ollama model after the discovery's last call."""
        self.calls.release()


def make_mapper(
    rater: Any, context: str | None = None, calls: VisionCalls | None = None
) -> HudMapper:
    """A :class:`HudMapper` asking the shared vision client ``rater`` through ``calls`` (one
    shared with :func:`cutpoint.events.feed_labels.make_labeler`; a new one when None), with the
    user's ``--context`` words (``settings.context``) as a hint."""
    return HudMapper(rater, context, calls or VisionCalls(rater))
