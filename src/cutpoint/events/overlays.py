"""Vision points, OCR reads: the on-screen overlays of non-gameplay footage (#1069 (c)).

Gameplay has its HUD map (:mod:`cutpoint.events.hud_map`). Other footage can carry text that
marks strong moments too: a scoreboard or score banner (sport, action), gauges such as speed or
altitude (action footage) and lower-thirds naming who speaks (talk). Which kinds are looked for
follows the project's style (:func:`cutpoint.events.layers.overlay_kinds`); sport is no category
of its own: a scoreboard counts once OCR confirms one.

* **Samples.** :data:`SAMPLE_FRAMES` frames spread over the footage (each source by its length)
  are grabbed and OCR'd whole (cached per source in ``<source cache>/overlays``).
* **Vision points.** Up to :data:`MAP_FRAMES` of them, the ones with the most text, are shown to
  the shared vision model (the one ``cutpoint events`` and the HUD map use: Ollama, or a hosted
  Vision model the user allowed; never without the Vision consent), one call each within
  :data:`MAP_BUDGET` seconds, with a generic prompt naming the wanted kinds (no game, sport or
  activity words; the user's ``--context`` as a hint). It points: kinds and boxes, nothing more.
  It sees the frames only, no file names or paths. Its words are logged at DEBUG only. An answer
  is cached per frame, model, prompt, kinds and ``--context``.
* **OCR confirms.** The boxes of all answers are grouped (one kind, overlapping boxes: the union
  of what the model saw in each frame) and each group counts only when the OCR of the samples
  reads text of its kind in it (:func:`judge`): a scoreboard small score numbers, a gauge
  numbers that change, a lower-third a line of words that comes and goes. A group the model
  marked in one frame needs more OCR evidence (:data:`STRONG_FRAMES`).
* **OCR reads.** The confirmed boxes are read through the footage (one keyframe-only decode of
  their area per source, at most one read every :data:`READ_STEP_S`, OCR only when the pixels
  changed), cached per source and boxes.
* **Events** (``kind`` ``ocr``, flagged ``overlay``): a scoreboard's score changes as the HUD's
  score readout makes them, under the overlay rules (a change of 1 to 6 points, up to 3
  digits: :data:`~cutpoint.events.score_readout.OVERLAY_SCORE_RULES`; ``score`` events), a gauge's peaks (:func:`gauge_peaks`, ``gauge_peak``) and a lower-third coming on
  screen (:func:`lower_thirds`, ``lower_third``). They add points through the event types
  (:data:`~cutpoint.events.eventtypes.OVERLAY_TYPES`): boost-only and at most
  :data:`~cutpoint.events.eventtypes.OVERLAY_MAX_POINTS` on a moment. The vision model never
  scores or names a moment.

Without the events extra the layer does not run; without a vision model (or the consent) it
asks nothing and the analysis is as before, but the overlay events of an earlier run with the
same kinds are kept (:data:`RESULT_FILE`).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, progress
from cutpoint.events import hud_map
from cutpoint.events.discover import (
    REF_W,
    SCORE_SLOT_PX,
    _alpha_share,
    _lines,
    _norm_text,
    boxes_in,
    score_value,
)
from cutpoint.events.feed_labels import STOP_AFTER, StopLabelling, model_key
from cutpoint.events.regions import REF_H, clamp_box, letters
from cutpoint.events.score_readout import OVERLAY_SCORE_RULES, ScoreTracker
from cutpoint.utils import context_block, context_hash

logger = logging.getLogger(__name__)

OVERLAY_VERSION = 2  # bump when the prompt, schema, samples or read change: caches not reused
FOLDER = "overlays"  # in each source's cache folder
SAMPLES_FILE = "samples.json"  # sample frames' OCR
VISION_FILE = "vision.json"  # the vision model's answers
RESULT_FILE = "result.json"  # the last run's overlay events of the source
SAMPLE_FRAMES = 12  # sample frames over the whole footage...
SAMPLE_MAX = 24  # ...at least one per source, but no more than this
SAMPLE_SPAN = (0.1, 0.9)  # share of a source the samples are spread over
SAMPLE_WIDTH = 1280  # px of a sample frame (its height keeps the aspect)
MIN_SCORE = 0.6  # OCR text below this recognition score is dropped (as the HUD OCR)
MAP_FRAMES = 3  # frames shown to the vision model, one call each
MAP_BUDGET = 60  # seconds of vision calls per footage
NUM_PREDICT = 700  # reply tokens: a few elements
PAD = 8  # reference px around the OCR boxes of a confirmed overlay
# an overlay marked in one frame only counts when OCR backs it in this many sample frames
STRONG_FRAMES = {"scoreboard": 5, "gauge": 5, "lower_third": 2}
MAX_AREA = {"scoreboard": 0.08, "gauge": 0.05, "lower_third": 0.2}  # share of the frame
SCORE_MIN_FRAMES = 3  # scoreboard: a score number (score_value) in this many sample frames
GAUGE_MIN_FRAMES = 3  # gauge: a number (gauge_value) in this many sample frames...
GAUGE_MIN_VALUES = 2  # ...with this many values...
GAUGE_MAX_ALPHA = 0.5  # ...and mostly digits
LT_MIN_LETTERS = 4  # lower-third: a line of at least this many letters...
LT_MIN_ALPHA = 0.6  # ...mostly letters...
LT_MIN_FRAMES = 1  # ...in this many sample frames...
LT_MAX_SHARE = 0.6  # ...but not in more than this share of them (a title that stays)
# the read: one keyframe-only decode; a frame at most every READ_STEP_S is looked at, and OCR'd
# only when the overlay's pixels changed since its last read (the HUD OCR's det skip measure)
READ_STEP_S = 1.0
READ_MIN_H = 64  # px: a smaller crop is upscaled to this height for OCR
CHANGE_SCALE = 4
CHANGE_TILE = 8
CHANGE_MAX_DIFF = 8.0
# gauge peaks (gauge_peaks): a read counts when another within GAUGE_RUN_S agrees within
# GAUGE_RUN_TOL (a misread digit is no peak); a gauge with fewer than GAUGE_MIN_READS reads or
# GAUGE_MIN_DISTINCT values does not move: no peaks. A peak is a counted read at least the
# GAUGE_TOP quantile of the source's reads that is the highest within GAUGE_WINDOW_S and stands
# GAUGE_RISE of the gauge's range (5th to 95th percentile) above the lowest read there; peaks
# are GAUGE_GAP_S apart at least, the highest kept
GAUGE_RUN_S = 3.0
GAUGE_RUN_TOL = 0.2
GAUGE_MIN_READS = 20
GAUGE_MIN_DISTINCT = 5
GAUGE_TOP = 0.9
GAUGE_WINDOW_S = 20.0
GAUGE_RISE = 0.15
GAUGE_GAP_S = 30.0
# lower-thirds (lower_thirds): a line of words (LT_MIN_LETTERS, LT_MIN_ALPHA) read in
# LT_CONFIRM reads in a row is on screen; it comes on screen after LT_GAP_S without it (or with
# other words). A box with words in more than LT_MAX_READ_SHARE of its reads is a title that
# stays: no events
LT_CONFIRM = 2
LT_GAP_S = 8.0
LT_SAME = 0.7  # share of letters two reads share at least to be the same words
LT_MAX_READ_SHARE = 0.6
LT_TEXT_MAX = 60  # characters of a lower-third's words kept in its event

IGNORE_KINDS = ("clock", "logo", "caption", "other")
# per overlay kind its definition in the prompt: generic, no game, sport or activity words
_WANTED_DEFS = {
    "scoreboard": (
        "- scoreboard: a scoreboard or score banner: the points, goals or rounds each side or "
        "competitor has.\n"
    ),
    "gauge": (
        "- gauge: a number overlaid on the picture that measures the activity, such as speed, "
        "altitude, distance or heart rate, often with its unit.\n"
    ),
    "lower_third": (
        "- lower_third: a caption naming who is on screen or speaking, or the topic: a name, a "
        "title or a role, usually near the bottom of the picture.\n"
    ),
}
_IGNORE_DEFS = (
    "- clock: a clock, a timer or a date.\n"
    "- logo: a channel logo, a watermark or a title that stays on screen.\n"
    "- caption: subtitles of what is said.\n"
    "- other: any other text.\n"
)
_PROMPT = (
    "This is one frame of a recorded video. Point out where these overlays are on screen, so "
    "that text recognition can read them later:\n"
    "{wanted}"
    "Also mark these, so that they are ignored:\n"
    "{ignore}"
    "{context}"
    "Only point: mark what is visible in this frame and leave out the rest (no element when "
    "there is none); do not judge the moment. box: [x0, y0, x1, y1] of the element on a 0-1000 "
    "scale of the frame's width and height. text_seen: a few words of the text in the box, or "
    "empty."
)
# A gauge number (:func:`gauge_value`). Thousands separators: a ",", "." or one space
# followed by exactly 3 digits, after 1-3 leading digits that do not start with 0, is a
# separator ("1,234 m", "1 234", "1.234", "12,345.6"); the same separator between every
# group. Any other "," or "." before 1-2 digits is the decimal mark ("12.5", "12,5").
# So "1.234" is 1234 and "0.125" is no value (3 decimals: a gauge shows 1 or 2).
_GAUGE_RE = re.compile(
    r"[^\d:%]{0,6}?"
    r"(?P<int>-?(?:[1-9]\d{0,2}(?P<sep>[ ,.])\d{3}(?:(?P=sep)\d{3})*|\d{1,5}))"
    r"(?:[.,](?P<dec>\d{1,2}))?"
    r"[^\d:%]{0,6}"
)

# A scoreboard's two numbers in one OCR box ("2-1", "LIV 2 - 1 MCI"): split into two items
# at the box centre -/+ a quarter of its width (at least SPLIT_MIN_PX apart: two score slots).
_SCORE_PAIR_RE = re.compile(r"^\D{0,4}?(\d{1,3})\s*[-\u2013]\s*(\d{1,3})\D{0,4}$")
SPLIT_MIN_PX = SCORE_SLOT_PX / 2 + 1

Rect = tuple[int, int, int, int]  # x, y, w, h in reference px
Ask = Callable[..., dict[str, Any] | None]


def schema(kinds: Sequence[str]) -> dict[str, Any]:
    """The reply schema of the map for the wanted ``kinds`` (the HUD map's, other kinds)."""
    return {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": [*kinds, *IGNORE_KINDS]},
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


def prompt(kinds: Sequence[str], context: str | None = None) -> str:
    """The map's prompt for the wanted ``kinds``, with the user's ``--context`` words as a hint."""
    return _PROMPT.format(
        wanted="".join(_WANTED_DEFS[k] for k in kinds),
        ignore=_IGNORE_DEFS,
        context=context_block(context),
    )


def gauge_value(text: str) -> float | None:
    """The number a gauge shows ("42", "12.5 km/h", "1043m", "1,234 m"; separators at
    :data:`_GAUGE_RE`), else None; never with ":" or "%" (a clock, a share such as a
    battery)."""
    m = _GAUGE_RE.fullmatch(text.strip())
    if m is None:
        return None
    whole = m["int"].replace(m["sep"], "") if m["sep"] else m["int"]
    return float(f"{whole}.{m['dec']}" if m["dec"] else whole)


def _words(text: str) -> str | None:
    """The words of a lower-third line (LT_MIN_LETTERS letters, mostly letters), else None."""
    t = " ".join(text.split())
    if len(letters(t)) < LT_MIN_LETTERS or _alpha_share([t.replace(" ", "")]) < LT_MIN_ALPHA:
        return None
    return t


# --------------------------------------------------------------------------- samples


def sample_plan(sources: Sequence[Any]) -> list[tuple[Any, float]]:
    """``(source, time)`` of the sample frames: :data:`SAMPLE_FRAMES` over the footage, each
    source a share by its length (at least one, :data:`SAMPLE_MAX` in all), spread over
    :data:`SAMPLE_SPAN` of it."""
    srcs = [s for s in sources if s.duration and s.duration > 0]
    total = sum(s.duration for s in srcs)
    if not srcs or total <= 0:
        return []
    out = []
    for s in srcs[:SAMPLE_MAX]:
        n = max(1, round(SAMPLE_FRAMES * s.duration / total))
        a, b = SAMPLE_SPAN[0] * s.duration, SAMPLE_SPAN[1] * s.duration
        out += [(s, round(a + (b - a) * (i + 0.5) / n, 2)) for i in range(n)]
    return out[:SAMPLE_MAX]


def _sample_key() -> dict[str, Any]:
    from cutpoint.events.ocr_raw import engine_version  # noqa: PLC0415

    return {"v": OVERLAY_VERSION, "width": SAMPLE_WIDTH, "engine": engine_version()}


def _to_boxes(items: list[tuple[list, str, float]], w: int, h: int) -> list[dict[str, Any]]:
    """OCR items of a ``w`` x ``h`` image as boxes in reference px (x and y scale apart)."""
    sx, sy = REF_W / w, REF_H / h
    out = []
    for box, text, score in items:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append(
            {
                "x0": round(min(xs) * sx),
                "y0": round(min(ys) * sy),
                "x1": round(max(xs) * sx),
                "y1": round(max(ys) * sy),
                "text": str(text),
                "score": round(float(score), 3),
            }
        )
    return out


def _stored_boxes(boxes: Any) -> list[dict[str, Any]] | None:
    """Cached boxes, checked (a hand-edited or older file never crashes)."""
    if not isinstance(boxes, list):
        return None
    out = []
    for b in boxes:
        if not isinstance(b, dict) or not isinstance(b.get("text"), str):
            return None
        if not all(isinstance(b.get(k), int | float) for k in ("x0", "y0", "x1", "y1")):
            return None
        out.append(b)
    return out


def ocr_samples(
    plan: list[tuple[Any, float]], make_engine: Callable[[], Any], force: bool = False
) -> list[dict[str, Any]]:
    """The sample frames of ``plan``, grabbed and OCR'd whole (cached per source in
    :data:`SAMPLES_FILE`, the JPEG next to it): ``{"id", "source", "t", "frame" (its JPEG),
    "size", "boxes"}``, the boxes in reference px. A frame that cannot be grabbed is left out."""
    from cutpoint.errors import FFmpegError  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import grab_images  # noqa: PLC0415
    from cutpoint.imageio import read_image  # noqa: PLC0415

    key = cache.params_hash(_sample_key(), 12)
    stores: dict[Path, dict[str, Any]] = {}
    out = []
    for s, t in plan:
        folder = Path(s.cdir) / FOLDER
        path = folder / SAMPLES_FILE
        if path not in stores:
            loaded = None if force else cache.load_json(path)
            stores[path] = loaded if isinstance(loaded, dict) and loaded.get("key") == key else {}
            stores[path]["key"] = key
            stores[path].setdefault("frames", {})
        store = stores[path]
        fid = f"{t:.2f}"
        name = f"sample-{key}-{fid.replace('.', '_')}.jpg"
        hit = store["frames"].get(fid)
        boxes = _stored_boxes(hit.get("boxes")) if isinstance(hit, dict) else None
        if boxes is not None and (folder / name).is_file():
            size = hit.get("size")
            if isinstance(size, list) and len(size) == 2:
                out.append(_sample(s, t, folder / name, tuple(size), boxes))
                continue
        try:
            jpeg = grab_images(s.io_path, t, width=SAMPLE_WIDTH, quality=3, keyframe=True)[0]
        except (FFmpegError, OSError) as err:
            logger.debug("Overlays: no sample frame at %.1fs of %s: %s", t, s.name, err)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(jpeg)
        img = read_image(folder / name)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = _to_boxes(make_engine()(img, MIN_SCORE), w, h)
        store["frames"][fid] = {"size": [w, h], "boxes": boxes}
        cache.save_json(path, store)
        out.append(_sample(s, t, folder / name, (w, h), boxes))
    return out


def _sample(s: Any, t: float, jpeg: Path, size: tuple[int, int], boxes: list) -> dict[str, Any]:
    return {
        "id": f"{s.name}@{t:.2f}",
        "source": s,
        "t": t,
        "frame": jpeg,
        "size": (int(size[0]), int(size[1])),
        "boxes": boxes,
    }


def pick_frames(samples: list[dict[str, Any]], n: int = MAP_FRAMES) -> list[dict[str, Any]]:
    """Up to ``n`` sample frames to show: those with the most text, a source not shown yet
    first; frames without text are never shown (nothing to point at)."""
    pool = sorted((f for f in samples if f["boxes"]), key=lambda f: (-len(f["boxes"]), f["id"]))
    chosen: list[dict[str, Any]] = []
    for f in pool:
        if len(chosen) == n:
            break
        if all(c["source"] is not f["source"] for c in chosen):
            chosen.append(f)
    for f in pool:
        if len(chosen) == n:
            break
        if f not in chosen:
            chosen.append(f)
    return chosen


# --------------------------------------------------------------------------- vision points


def _key(model: str, frame: dict[str, Any], kinds: Sequence[str], context: str | None) -> str:
    return cache.params_hash(
        {
            "v": OVERLAY_VERSION,
            "model": model,
            "prompt": cache.params_hash(
                {"p": _PROMPT, "w": _WANTED_DEFS, "i": _IGNORE_DEFS, "s": schema(kinds)}, 8
            ),
            "frame": f"{frame['t']:.2f}|{frame['frame'].name}",
            "kinds": list(kinds),
            "context": context_hash(context),
        },
        16,
    )


def _stored(els: Any, kinds: Sequence[str]) -> list[dict[str, Any]] | None:
    """Cached elements of ``kinds``, checked (a hand-edited or older file never crashes)."""
    if not isinstance(els, list):
        return None
    out = []
    for e in els:
        if isinstance(e, dict) and e.get("kind") in kinds:
            box = e.get("box")
            if isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box):
                got = clamp_box(*box)
                if got is not None:
                    out.append({"kind": e["kind"], "box": list(got)})
    return out


def point(
    frames: list[dict[str, Any]],
    kinds: Sequence[str],
    ask: Ask,
    model: str,
    context: str | None = None,
    force: bool = False,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Ask the vision model where the overlays of ``kinds`` are on each of ``frames`` (cached
    per frame in its source's :data:`VISION_FILE`). Returns the elements per frame answered
    (:func:`cutpoint.events.hud_map.parse`) and the calls made. ``ask``: ``(prompt, jpegs,
    schema, num_predict) -> reply or None``; it may raise :class:`StopLabelling`."""
    allowed = (*kinds, *IGNORE_KINDS)
    text, sch = prompt(kinds, context), schema(kinds)
    seen: list[list[dict[str, Any]]] = []
    calls = failed = 0
    for f in frames:
        path = Path(f["source"].cdir) / FOLDER / VISION_FILE
        loaded = cache.load_json(path)
        store: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        key = _key(model, f, kinds, context)
        hit = None if force else store.get(key)
        els = _stored(hit.get("elements"), allowed) if isinstance(hit, dict) else None
        if els is None:
            try:
                reply = ask(text, [f["frame"].read_bytes()], sch, NUM_PREDICT)
            except StopLabelling as why:
                logger.info("Overlays: %s; going on with the frames answered", why)
                break
            calls += 1
            els = hud_map.parse(reply, f["size"], allowed)
            if els is None:
                failed += 1
                if failed >= STOP_AFTER:
                    logger.info("Overlays: the vision model gave no usable answer; stopping")
                    break
                continue
            failed = 0
            store[key] = {"elements": els}
            path.parent.mkdir(parents=True, exist_ok=True)
            cache.save_json(path, store)
        seen.append(els)
    return seen, calls


# --------------------------------------------------------------------------- OCR confirms


def score_numbers(text: str) -> list[int]:
    """The score numbers of one scoreboard OCR box: both of a pair ("2-1",
    :data:`_SCORE_PAIR_RE`), else its one number (:func:`score_value`, overlay digits), else
    none."""
    m = _SCORE_PAIR_RE.match(text.strip())
    if m:
        return [int(m[1]), int(m[2])]
    v = score_value(text, OVERLAY_SCORE_RULES.digits)
    return [] if v is None else [v]


def _in_rect(x: float, y: float | None, r: Sequence[int]) -> bool:
    """Whether a centre lies in the rectangle (a centre without its y: its x alone)."""
    return r[0] <= x <= r[0] + r[2] and (y is None or r[1] <= y <= r[1] + r[3])


def ocr_stats(
    frames: list[dict[str, Any]], zone: Rect, clocks: Sequence[Rect] = ()
) -> dict[str, Any]:
    """What the samples' OCR read in ``zone`` (counts only): frames with text and their share,
    frames with a score number and its values (no box centred in one of ``clocks``, the grown
    boxes of the clocks the model marked: a match minute is no score), frames with a gauge
    number and its values, frames with a line of words, the letter share."""
    x, y, w, h = zone
    per = [boxes_in(f, x, y, w, h) for f in frames]
    inside = [bs for bs in per if bs]
    texts = [b["text"] for bs in inside for b in bs]

    def clock(b: dict[str, Any]) -> bool:
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        return any(_in_rect(cx, cy, c) for c in clocks)

    scores = [[b for b in bs if score_numbers(b["text"]) and not clock(b)] for bs in inside]
    gauges = [[b for b in bs if gauge_value(b["text"]) is not None] for bs in inside]
    words = [[ln for ln in _lines(bs) if _words(" ".join(b["text"] for b in ln))] for bs in inside]
    return {
        "frames": len(inside),
        "share": round(len(inside) / max(1, len(frames)), 3),
        "alpha": round(_alpha_share(texts), 3),
        "score_frames": sum(bool(bs) for bs in scores),
        "score_values": sorted({v for bs in scores for b in bs for v in score_numbers(b["text"])}),
        "gauge_frames": sum(bool(bs) for bs in gauges),
        "gauge_values": len({gauge_value(b["text"]) for bs in gauges for b in bs}),
        "word_frames": sum(bool(ls) for ls in words),
        "_boxes": {
            "scoreboard": [b for bs in scores for b in bs],
            "gauge": [b for bs in gauges for b in bs],
            "lower_third": [b for ls in words for ln in ls for b in ln],
        },
    }


def _test(kind: str, st: dict[str, Any]) -> str | None:
    """Why OCR does not confirm an overlay of ``kind`` (None: it does)."""
    if kind == "scoreboard":
        if st["score_frames"] < SCORE_MIN_FRAMES:
            return "OCR reads no score numbers"
        return None
    if kind == "gauge":
        if st["gauge_frames"] < GAUGE_MIN_FRAMES or st["gauge_values"] < GAUGE_MIN_VALUES:
            return "OCR reads no numbers that change"
        if st["alpha"] > GAUGE_MAX_ALPHA:
            return "OCR reads words, no gauge"
        return None
    if kind == "lower_third":
        if st["word_frames"] < LT_MIN_FRAMES:
            return "OCR reads no words"
        if st["share"] > LT_MAX_SHARE:
            return "text there most of the time, no lower-third"
        return None
    return "not a wanted kind"


def judge(
    g: dict[str, Any], frames: list[dict[str, Any]], clocks: Sequence[Rect] = ()
) -> dict[str, Any]:
    """The record of an element group ``g`` (:func:`cutpoint.events.hud_map.group`):
    ``confirmed`` and its rectangle ``rect`` (reference px: the OCR boxes of its kind inside
    the grown box, padded) when the samples' OCR confirms it (:func:`_test`), else ``why``
    not. ``clocks``: the grown boxes of the clocks marked (:func:`map_overlays`); a scoreboard
    reads no number inside them and keeps those it overlaps as ``clocks``."""
    kind, n = g["kind"], len(g["frames"])
    rec: dict[str, Any] = {"kind": kind, "box": list(g["box"]), "seen": n, "confirmed": False}
    if kind not in _WANTED_DEFS:
        rec["why"] = "ignored kind"
        return rec
    zone = hud_map._grow(g["box"])
    clocks = clocks if kind == "scoreboard" else ()
    st = ocr_stats(frames, zone, clocks)
    rec["evidence"] = {k: v for k, v in st.items() if not k.startswith("_")}
    why = _test(kind, st)
    if why is None and n < 2 and st["frames"] < STRONG_FRAMES[kind]:
        why = "marked in one frame, OCR too weak"
    bs = st["_boxes"][kind]
    if why is None and not bs:
        why = "no OCR box inside"
    if why is not None:
        rec["why"] = why
        return rec
    x0 = max(zone[0], min(b["x0"] for b in bs))
    y0 = max(zone[1], min(b["y0"] for b in bs))
    x1 = min(zone[0] + zone[2], max(b["x1"] for b in bs))
    y1 = min(zone[1] + zone[3], max(b["y1"] for b in bs))
    rect = clamp_box(x0 - PAD, y0 - PAD, x1 - x0 + 2 * PAD, y1 - y0 + 2 * PAD)
    if rect is None:
        rec["why"] = "no OCR box inside"
        return rec
    if rect[2] * rect[3] > MAX_AREA[kind] * REF_W * REF_H:
        rec["why"] = "too large"
        return rec
    rec |= {"confirmed": True, "rect": list(rect)}
    near = [list(c) for c in clocks if _overlaps(rect, c)]
    if near:
        rec["clocks"] = near
    return rec


def _overlaps(a: Sequence[int], b: Sequence[int]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def regions(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The confirmed overlays of ``elements`` (:func:`judge`) to read: ``{"name", "kind",
    "rect"}``, one per place (an overlay of one kind over another's box is left out)."""
    out: list[dict[str, Any]] = []
    taken: Counter[str] = Counter()
    for e in elements:
        if not e.get("confirmed"):
            continue
        rect = tuple(e["rect"])
        if any(hud_map.iou(rect, tuple(r["rect"])) >= hud_map.MATCH_IOU for r in out):
            continue
        taken[e["kind"]] += 1
        reg = {"name": f"{e['kind']}{taken[e['kind']]}", "kind": e["kind"], "rect": list(rect)}
        if e.get("clocks"):
            reg["clocks"] = [list(c) for c in e["clocks"]]
        out.append(reg)
    return out


def map_overlays(
    samples: list[dict[str, Any]],
    kinds: Sequence[str],
    ask: Ask,
    model: str,
    context: str | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    """Vision points at the overlays on up to :data:`MAP_FRAMES` sample frames, OCR of all
    ``samples`` confirms them. Returns ``{"model", "frames" (answered), "calls", "elements"}``
    (each as :func:`judge` makes it), or None when the model answered for no frame."""
    shown = pick_frames(samples)
    if not shown:
        return None
    seen, calls = point(shown, kinds, ask, model, context, force)
    if not seen:
        return None
    groups = hud_map.group(seen)
    clocks = [hud_map._grow(g["box"]) for g in groups if g["kind"] == "clock"]
    elements = [judge(g, samples, clocks) for g in groups]
    return {"model": model, "frames": len(seen), "calls": calls, "elements": elements}


def describe(elements: list[dict[str, Any]]) -> str:
    """The map's elements in one line (kinds, rectangles and verdicts; no text)."""
    parts = []
    for e in elements:
        if e["kind"] not in _WANTED_DEFS:
            continue
        where = "{}@{},{} {}x{}".format(e["kind"], *(e.get("rect") or e["box"]))
        verdict = "confirmed" if e.get("confirmed") else f"not kept: {e.get('why', '?')}"
        parts.append(f"{where} ({e['seen']} frame(s), {verdict})")
    return ", ".join(parts) or "nothing marked"


# --------------------------------------------------------------------------- OCR reads


def _px(rect: Sequence[int], width: int, height: int) -> Rect:
    """A reference rectangle in a ``width`` x ``height`` frame (even values, inside it)."""
    sx, sy = width / REF_W, height / REF_H
    x, y = int(rect[0] * sx) // 2 * 2, int(rect[1] * sy) // 2 * 2
    w = min(max(2, int(rect[2] * sx) // 2 * 2), width - x)
    h = min(max(2, int(rect[3] * sy) // 2 * 2), height - y)
    return x, y, w, h


def _change_image(crop: np.ndarray) -> np.ndarray:
    import cv2  # noqa: PLC0415

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    size = (max(1, w // CHANGE_SCALE), max(1, h // CHANGE_SCALE))
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA)


def _changed(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two :func:`_change_image` images differ in some :data:`CHANGE_TILE` px tile by
    :data:`CHANGE_MAX_DIFF` (mean abs difference, 0-255) or more."""
    if a.shape != b.shape:
        return True
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    ys = np.arange(0, d.shape[0], CHANGE_TILE)
    xs = np.arange(0, d.shape[1], CHANGE_TILE)
    sums = np.add.reduceat(np.add.reduceat(d, ys, axis=0), xs, axis=1)
    counts = np.outer(np.diff(ys, append=d.shape[0]), np.diff(xs, append=d.shape[1]))
    return float((sums / counts).max()) >= CHANGE_MAX_DIFF


def _ocr_crop(
    engine: Any, crop: np.ndarray, sx: float, ox: float, sy: float = 0.0, oy: float = 0.0
) -> list[list[Any]]:
    """OCR of one overlay crop: ``[text, score, centre x, width, centre y]`` per box, in
    reference px (``sx``/``sy``: crop px to reference px, ``ox``/``oy``: the crop's left and
    top edge in reference px)."""
    scale = 1.0
    if crop.shape[0] < READ_MIN_H:
        import cv2  # noqa: PLC0415

        scale = READ_MIN_H / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = []
    for box, text, score in engine(crop, MIN_SCORE):
        x0, x1 = min(p[0] for p in box) / scale, max(p[0] for p in box) / scale
        cy = (min(p[1] for p in box) + max(p[1] for p in box)) / 2 / scale
        out.append(
            [
                str(text),
                round(float(score), 3),
                round(ox + (x0 + x1) / 2 * sx, 1),
                round((x1 - x0) * sx, 1),
                round(oy + cy * sy, 1),
            ]
        )
    return out


def _read_key(s: Any, regs: list[dict[str, Any]], avoid: Sequence[tuple[float, float]]) -> dict:
    from cutpoint.events.ocr_raw import engine_version  # noqa: PLC0415

    return {
        "v": OVERLAY_VERSION,
        "regions": [[r["name"], r["kind"], list(r["rect"])] for r in regs],
        "step": READ_STEP_S,
        "engine": engine_version(),
        "avoid": [list(a) for a in avoid],
    }


def read_source(
    s: Any,
    regs: list[dict[str, Any]],
    make_engine: Callable[[], Any],
    *,
    hwaccel: bool = True,
    avoid: Sequence[tuple[float, float]] = (),
    force: bool = False,
) -> dict[str, list[list[Any]]]:
    """The reads of the overlays ``regs`` (:func:`regions`) through source ``s``: per region
    ``[[t, [[text, score, centre x, width, centre y], ...]], ...]`` (cached per source and regions). One
    keyframe-only decode of their area; a frame at most every :data:`READ_STEP_S`, a region
    OCR'd only when its pixels changed (else its last read again); reads in ``avoid`` ranges
    (unreadable parts) are left out."""
    from cutpoint.events.common import layer_cache_path  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import decode_raw, find_tool, probe  # noqa: PLC0415

    path = layer_cache_path(Path(s.cdir), "overlays", _read_key(s, regs, avoid))
    hit = None if force else cache.load_json(path)
    if isinstance(hit, dict) and isinstance(hit.get("reads"), dict):
        return {str(k): v for k, v in hit["reads"].items() if isinstance(v, list)}
    find_tool("ffmpeg")
    info = probe(s.io_path)
    width, height = info.width, info.height
    rects = {r["name"]: _px(r["rect"], width, height) for r in regs}
    ux0 = min(x for x, _, _, _ in rects.values())
    uy0 = min(y for _, y, _, _ in rects.values())
    ux1 = max(x + w for x, _, w, _ in rects.values())
    uy1 = max(y + h for _, y, _, h in rects.values())
    cw, ch = ux1 - ux0, uy1 - uy0
    sx, sy = REF_W / width, REF_H / height
    engine = make_engine()
    reads: dict[str, list[list[Any]]] = {r["name"]: [] for r in regs}
    last: dict[str, tuple[np.ndarray, list[list[Any]]]] = {}
    next_t = -1.0
    frames = decode_raw(
        s.io_path,
        size=(cw, ch),
        vf=f"crop={cw}:{ch}:{ux0}:{uy0}",
        pix_fmt="bgr24",
        keyframes_only=True,
        pts=True,
        hwaccel="auto" if hwaccel else None,
    )
    try:
        for t, frame in frames:
            if t < next_t or any(a <= t <= b for a, b in avoid):
                continue
            next_t = t + READ_STEP_S
            progress.update(detail=f"reading on-screen overlays: {s.name} {t / 60:.0f} min")
            for name, (x, y, w, h) in rects.items():
                crop = np.ascontiguousarray(frame[y - uy0 : y - uy0 + h, x - ux0 : x - ux0 + w])
                img = _change_image(crop)
                before = last.get(name)
                if before is not None and not _changed(before[0], img):
                    items = before[1]
                else:
                    items = _ocr_crop(engine, crop, sx, x * sx, sy, y * sy)
                    last[name] = (img, items)
                reads[name].append([round(float(t), 2), items])
    finally:
        frames.close()
    cache.save_json(path, {"reads": reads}, indent=0)
    return reads


# --------------------------------------------------------------------------- events


def _score_items(
    items: list[list[Any]], clocks: Sequence[Sequence[int]]
) -> list[tuple[list, str, float]]:
    """A scoreboard read's items for :meth:`ScoreTracker.see`: boxes centred in one of
    ``clocks`` left out (a match minute), a pair ("2-1") split into its two numbers at the box
    centre -/+ a quarter of its width (:data:`SPLIT_MIN_PX` at least)."""
    out = []
    for text, score, cx, *more in items:
        w = more[0] if more else 0.0
        cy = more[1] if len(more) > 1 else None
        if any(_in_rect(cx, cy, c) for c in clocks):
            continue
        m = _SCORE_PAIR_RE.match(str(text).strip())
        if m:
            off = max(w / 4, SPLIT_MIN_PX)
            out += [([[cx - off, 0.0]] * 2, m[1], score), ([[cx + off, 0.0]] * 2, m[2], score)]
        else:
            out.append(([[cx, 0.0]] * 2, text, score))
    return out


def score_changes(
    name: str, reads: list[list[Any]], clocks: Sequence[Sequence[int]] = ()
) -> tuple[list[dict[str, Any]], dict]:
    """The ``score`` events of a scoreboard's reads, by the HUD's score readout rules under
    the overlay rules (:class:`~cutpoint.events.score_readout.ScoreTracker`,
    :data:`~cutpoint.events.score_readout.OVERLAY_SCORE_RULES`), and its per-slot stats.
    ``clocks``: the clock boxes inside it (:func:`judge`), never read as scores."""
    tracker = ScoreTracker(OVERLAY_SCORE_RULES)
    for t, items in reads:
        tracker.see(float(t), name, _score_items(items, clocks), (0.0, 0.0, 1.0, 1.0))
    return tracker.finish(set())


def _counted(reads: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """The gauge reads ``(t, value, score)`` another read within :data:`GAUGE_RUN_S` agrees
    with (within :data:`GAUGE_RUN_TOL`)."""
    out = []
    for i, (t, v, c) in enumerate(reads):
        for j in (i - 1, i + 1):
            if 0 <= j < len(reads):
                t2, v2, _ = reads[j]
                if abs(t2 - t) <= GAUGE_RUN_S and abs(v2 - v) <= GAUGE_RUN_TOL * max(1.0, abs(v)):
                    out.append((t, v, c))
                    break
    return out


def gauge_peaks(name: str, reads: list[list[Any]]) -> tuple[list[dict[str, Any]], dict]:
    """The ``gauge_peak`` events of a gauge's reads (rules at :data:`GAUGE_RUN_S`) and its
    stats. A read's value is its surest number (:func:`gauge_value`)."""
    vals: list[tuple[float, float, float]] = []
    for t, items in reads:
        best = max(
            ((v, sc) for text, sc, *_ in items if (v := gauge_value(text)) is not None),
            key=lambda x: x[1],
            default=None,
        )
        if best is not None:
            vals.append((float(t), best[0], float(best[1])))
    counted = _counted(vals)
    stats: dict[str, Any] = {"reads": len(vals), "counted": len(counted), "peaks": 0}
    if len(counted) < GAUGE_MIN_READS or len({v for _, v, _ in counted}) < GAUGE_MIN_DISTINCT:
        stats["disabled"] = True
        return [], stats
    values = np.array([v for _, v, _ in counted])
    top = float(np.quantile(values, GAUGE_TOP))
    lo, hi = np.quantile(values, [0.05, 0.95])
    rise = GAUGE_RISE * float(hi - lo)
    found = []
    for t, v, c in counted:
        if v < top:
            continue
        near = [w for t2, w, _ in counted if abs(t2 - t) <= GAUGE_WINDOW_S]
        if v < max(near) or v - min(near) < rise or rise <= 0:
            continue
        found.append((v, t, c))
    kept: list[tuple[float, float, float]] = []
    for v, t, c in sorted(found, key=lambda x: (-x[0], x[1])):
        if all(abs(t - k[1]) >= GAUGE_GAP_S for k in kept):
            kept.append((v, t, c))
    events = [
        {
            "t": round(t, 2),
            "end": round(t, 2),
            "type": "gauge_peak",
            "text": f"{v:g}",
            "confidence": round(c, 3),
            "region": name,
            "value": v,
        }
        for v, t, c in sorted(kept, key=lambda x: x[1])
    ]
    stats["peaks"] = len(events)
    return events, stats


def _same_words(a: str, b: str) -> bool:
    from difflib import SequenceMatcher  # noqa: PLC0415

    x, y = _norm_text(a), _norm_text(b)
    return bool(x and y) and SequenceMatcher(None, x, y).ratio() >= LT_SAME


def lower_thirds(name: str, reads: list[list[Any]]) -> tuple[list[dict[str, Any]], dict]:
    """The ``lower_third`` events of a lower-third's reads (rules at :data:`LT_CONFIRM`): one
    when its words come on screen, ending at their last read, and its stats."""
    seen: list[tuple[float, str, float]] = []  # reads with words: (t, words, mean score)
    for t, items in reads:
        line = " ".join(text for text, *_ in sorted(items, key=lambda i: i[2]))
        words = _words(line)
        if words is not None:
            seen.append((float(t), words, float(np.mean([sc for _, sc, *_ in items]))))
    stats: dict[str, Any] = {"reads": len(reads), "with_words": len(seen), "shown": 0}
    if not reads or len(seen) > LT_MAX_READ_SHARE * len(reads):
        stats["disabled"] = bool(reads)
        return [], stats
    runs: list[list[tuple[float, str, float]]] = []
    for r in seen:
        last = runs[-1][-1] if runs else None
        if last is not None and r[0] - last[0] < LT_GAP_S and _same_words(r[1], last[1]):
            runs[-1].append(r)
        else:
            runs.append([r])
    events = []
    for run in runs:
        if len(run) < LT_CONFIRM:
            continue
        text = Counter(w for _, w, _ in run).most_common(1)[0][0]
        events.append(
            {
                "t": round(run[0][0], 2),
                "end": round(run[-1][0], 2),
                "type": "lower_third",
                "text": text[:LT_TEXT_MAX],
                "confidence": round(float(np.mean([c for _, _, c in run])), 3),
                "region": name,
            }
        )
    stats["shown"] = len(events)
    return events, stats


_READERS = {"scoreboard": score_changes, "gauge": gauge_peaks, "lower_third": lower_thirds}


def to_events(
    regs: list[dict[str, Any]], reads: dict[str, list[list[Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The overlay events of one source's ``reads`` (layer-1 OCR events flagged ``overlay``)
    and the stats per region."""
    events: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for r in regs:
        got_reads = reads.get(r["name"]) or []
        if r["kind"] == "scoreboard":
            got, st = score_changes(r["name"], got_reads, r.get("clocks") or ())
        else:
            got, st = _READERS[r["kind"]](r["name"], got_reads)
        stats[r["name"]] = st
        events += [e | {"kind": "ocr", "overlay": True, "overlay_kind": r["kind"]} for e in got]
    events.sort(key=lambda e: e["t"])
    return events, stats


# --------------------------------------------------------------------------- the layer


def _result_path(s: Any) -> Path:
    return Path(s.cdir) / FOLDER / RESULT_FILE


def kept_events(s: Any, kinds: Sequence[str]) -> list[dict[str, Any]] | None:
    """The overlay events an earlier run found in source ``s`` for the same ``kinds`` (None:
    none recorded)."""
    data = cache.load_json(_result_path(s))
    if not isinstance(data, dict) or data.get("v") != OVERLAY_VERSION:
        return None
    if data.get("kinds") != list(kinds) or not isinstance(data.get("events"), list):
        return None
    return [
        e
        for e in data["events"]
        if isinstance(e, dict) and isinstance(e.get("t"), int | float) and e.get("overlay")
    ]


def _keep(s: Any, kinds: Sequence[str], events: list[dict[str, Any]]) -> None:
    path = _result_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, {"v": OVERLAY_VERSION, "kinds": list(kinds), "events": events})


def _kept_all(sources: Sequence[Any], kinds: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for s in sources:
        got = kept_events(s, kinds)
        if got:
            out[str(s.path)] = got
    return out


def run_overlays(
    sources: Sequence[Any],
    kinds: Sequence[str],
    calls: Any,
    *,
    context: str | None = None,
    make_engine: Callable[[], Any],
    run: bool = True,
    force: bool = False,
    hwaccel: bool = True,
    avoid: dict[str, list[tuple[float, float]]] | None = None,
    vision: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """The overlay layer on ``sources``: the overlay events per source path and the stats.

    Args:
        sources: The project's sources (:class:`cutpoint.events.common.Source`).
        kinds: The overlay kinds of the project's style
            (:func:`cutpoint.events.layers.overlay_kinds`); none: nothing is done.
        calls: The shared vision client's calls
            (:class:`cutpoint.events.feed_labels.VisionCalls`), None without a vision client.
        context: The user's ``--context`` words, a hint in the prompt.
        make_engine: The OCR engine (built on the first OCR).
        run: False: the layer does not run; the events of an earlier run with these kinds are
            kept (:func:`kept_events`).
        force: Ignore the cached samples, answers and reads.
        hwaccel: Decode on the GPU when it works.
        avoid: Ranges per source path that are not read (unreadable parts).
        vision: False (``--no-vision``): no frame goes to the model; as without a vision
            client, the events of an earlier run are kept.
    """
    kinds = tuple(kinds)
    if not kinds:
        return {}, {}
    stats: dict[str, Any] = {"v": OVERLAY_VERSION, "kinds": list(kinds)}
    if not run:
        return _kept_all(sources, kinds), stats | {"skipped": "not run: earlier events kept"}
    if not vision:
        logger.info('Vision is off (--no-vision or "vision": false): overlays not read')
        return _kept_all(sources, kinds), stats | {"skipped": "vision off"}
    rater = getattr(calls, "rater", None)
    if calls is None or rater is None:
        return _kept_all(sources, kinds), stats | {"skipped": "no vision client"}
    if rater.consent_needed:
        logger.info("Overlays: the vision model needs your Vision consent; overlays not read")
        return _kept_all(sources, kinds), stats | {"skipped": "no vision consent"}
    if not calls.usable():
        return _kept_all(sources, kinds), stats | {"skipped": "no vision model answers"}
    progress.update(detail="finding on-screen overlays")
    samples = ocr_samples(sample_plan(sources), make_engine, force)
    stats["samples"] = len(samples)
    if not any(f["boxes"] for f in samples):
        logger.info("Overlays: no text on the sample frames; nothing to read")
        for s in sources:
            _keep(s, kinds, [])
        return {}, stats | {"elements": []}
    model = model_key(rater)
    calls.begin(("overlays", tuple(str(s.path) for s in sources)), MAP_BUDGET)
    try:
        found = map_overlays(samples, kinds, calls.ask, model, context, force)
    finally:
        calls.release()
    if found is None:
        logger.info("Overlays: the vision model answered for no frame; overlays not read")
        return _kept_all(sources, kinds), stats | {"skipped": "no answer"}
    logger.info("Overlays (%s): %s", model, describe(found["elements"]))
    regs = regions(found["elements"])
    stats |= {
        "model": model,
        "calls": found["calls"],
        "frames_shown": found["frames"],
        "elements": [
            {k: e[k] for k in ("kind", "box", "rect", "seen", "confirmed", "why") if k in e}
            for e in found["elements"]
            if e["kind"] in _WANTED_DEFS
        ],
        "regions": regs,
    }
    out: dict[str, list[dict[str, Any]]] = {}
    per_source: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    from cutpoint.events.common import SourceGuard  # noqa: PLC0415

    guard = SourceGuard("Overlays", log=logger)
    for s in sources:
        if not regs:
            _keep(s, kinds, [])
            continue
        try:
            reads = read_source(
                s,
                regs,
                make_engine,
                hwaccel=hwaccel,
                avoid=(avoid or {}).get(str(s.path), ()),
                force=force,
            )
        except Exception as err:  # noqa: BLE001 - one bad source (out of memory: SourceGuard)
            guard.skip(err, s.name, s.path)
            continue
        events, st = to_events(regs, reads)
        _keep(s, kinds, events)
        per_source[s.name] = st
        counts.update(e["type"] for e in events)
        if events:
            out[str(s.path)] = events
    stats["events"] = dict(sorted(counts.items()))
    if per_source:
        stats["sources"] = per_source
    if regs:
        logger.info(
            "Overlays: %s",
            ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "no events read",
        )
    return out, stats


def summary(stats: Any) -> dict[str, Any] | None:
    """What the overlay layer did, for ``settings.overlays`` of highlights.json (None when it
    did not run): its kinds, the model, the calls, the overlays confirmed and the events."""
    if not isinstance(stats, dict) or not stats.get("kinds"):
        return None
    out = {k: stats[k] for k in ("v", "kinds", "model", "calls", "events", "skipped") if k in stats}
    regs = stats.get("regions")
    if isinstance(regs, list):
        out["confirmed"] = [r.get("kind") for r in regs if isinstance(r, dict)]
    return out
