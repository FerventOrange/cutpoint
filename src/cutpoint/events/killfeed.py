"""Layer 1b: on-screen events (kill feed, round banners, "X VS Y") via OCR.

One sequential ffmpeg pass per source decodes only the bounding box of the
game's HUD regions at a low frame rate (``-hwaccel cuda`` when it works) and
pipes raw BGR frames to numpy. Each region is OCR'd with RapidOCR (CPU) only
when its "HUD pixel" mask (white text, blue/orange name boxes) has changed
since the last OCR of that region, which skips the vast majority of frames.

Events: ``{t, end, type, text, confidence, region, ...}`` with type one of
``kill``, ``banner``, ``vs``, ``points``, ``log``.
"""

from __future__ import annotations

import bisect
import difflib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.events import ocr_raw
from cutpoint.events.common import Source, layer_cache_path
from cutpoint.events.discover import DiscoverParams, _centre_cover, gameplay_frames
from cutpoint.events.regions import HudPreset, Region, game_key_for_path, load_preset, scaled
from cutpoint.ffmpeg_utils import READ_ERROR_RE, decode_raw, probe

logger = logging.getLogger(__name__)

# Your in-game names come from config.json ``player_tags`` (or ``--player``).
DEFAULT_PLAYER_TAGS: tuple[str, ...] = ()
_VS_RE = re.compile(r"(\d)\s*V\s*S\s*(\d)")
_POINTS_RE = re.compile(r"\+\s?(\d{2,4})\b")


@dataclass
class OcrParams:
    """Tunable parameters for :func:`detect_ocr_events` (all go into the cache key)."""

    fps: float = 3.0
    game: str = "auto"
    player_tags: tuple[str, ...] = DEFAULT_PLAYER_TAGS
    hwaccel: bool = True
    min_text_score: float = 0.6
    fuzzy: float = 0.8  # SequenceMatcher ratio for phrase / dedupe matches
    persist_s: float = 8.0  # an entry seen again within this is the same event
    regions: tuple[Region, ...] | None = None  # None = preset for the game
    extra: dict[str, Any] = field(default_factory=dict)
    project: Path | None = None  # for project-local presets (not part of the key)

    def key(self, game: str | None, preset: HudPreset | None = None) -> dict[str, Any]:
        """Cache key (hwaccel does not change results, so it is left out).

        Built-in presets (R6) produce the keys they had before JSON presets
        existed, but :func:`detect_ocr_events` adds the hash of the raw read's key
        (``"raw"``, which includes the RapidOCR version), so results cached before
        the raw OCR cache are read once more, and again after a RapidOCR upgrade.
        """
        regs = self.regions or (preset.all_regions() if preset else ())
        key = {
            "fps": self.fps,
            "game": game,
            "tags": list(self.player_tags),
            "min_text_score": self.min_text_score,
            "fuzzy": self.fuzzy,
            "persist_s": self.persist_s,
            "regions": [r.key() for r in regs],
            "extra": self.extra,
            "v": 3,
        }
        if preset is not None and preset.key():
            key["preset"] = preset.key() | {"interp": GENERIC_VERSION}
        return key


# bump when the generic (JSON preset) interpretation changes; 7: menu frames give no
# events (:func:`menu_times`), multi-word phrases and feed kill words match whole words only;
# 8: static screens are menus too (:func:`static_frames`); 9: menu frames keep feed sightings
# and kills, a static screen without new menu text must hold still STATIC_BARE_MIN_S;
# 10: new menu text is STATIC_TEXT_LINES texts that stay for STATIC_TEXT_SHARE of the part
GENERIC_VERSION = 10


# --------------------------------------------------------------------------- text utils


def norm(text: str) -> str:
    """Upper-case alphanumerics and single spaces (for fuzzy comparisons)."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", text.upper())).strip()


def fuzzy_contains(text: str, phrase: str, thr: float = 0.8) -> bool:
    """Whether ``phrase`` occurs in ``text`` allowing OCR typos.

    Words of ``text`` also end at camelCase boundaries ("TargetDestroyed" is
    "TARGET DESTROYED") and between letters and digits ("ROUND5").

    * phrases of <= 4 characters (e.g. "ACE") must equal a whole word;
    * other single-word phrases must fuzzy-match one word (or two adjacent
      words OCR split apart) of similar length, so "DEF ATK" is not "DEFEAT";
    * multi-word phrases are compared with the text with spaces removed,
      because OCR often drops spaces ("ReverseFriendlyFire hasbeen activated"),
      but only with a stretch of whole words (:func:`_word_spans`): it starts
      and ends at a word boundary or an end of the text and its length is
      within a fifth of the phrase's. So "Targetdestroyed" matches "TARGET
      DESTROYED", while "SHIELD UP" is not found inside "BLOCKEDSHIELDUPGRADE".
    """
    t, p = norm(_CAMEL_RE.sub(" ", text)), norm(phrase)
    if not t or not p:
        return False
    words = t.split()
    if len(p) <= 4:
        return p in words
    if " " not in p:
        cands = words + [a + b for a, b in pairwise(words)]
        return any(
            abs(len(w) - len(p)) <= max(1, len(p) // 5)
            and difflib.SequenceMatcher(None, w, p).ratio() >= max(thr, 0.85)
            for w in cands
        )
    if f" {p} " in f" {t} ":
        return True
    pc = p.replace(" ", "")
    m = difflib.SequenceMatcher(None, "", pc)  # the phrase's index is built once
    for span in _word_spans(t, len(pc), max(1, len(pc) // 5)):
        m.set_seq1(span)
        if m.quick_ratio() >= thr and m.ratio() >= thr:  # quick_ratio: an upper bound
            return True
    return False


_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")  # "TargetDestroyed": a word boundary


def _word_spans(t: str, n: int, tol: int) -> list[str]:
    """Space-less stretches of whole words of the normalized text ``t`` that are
    ``n`` +- ``tol`` characters long; letter and digit runs are words."""
    toks = re.findall(r"[A-Z]+|[0-9]+", t)
    bounds = [0]
    for tok in toks:
        bounds.append(bounds[-1] + len(tok))
    tc = "".join(toks)
    out = []
    for i, a in enumerate(bounds):
        for b in bounds[i + 1 :]:
            if b - a > n + tol:
                break
            if b - a >= n - tol:
                out.append(tc[a:b])
    return out


def similar(a: str, b: str) -> float:
    """SequenceMatcher ratio of two normalized strings."""
    return difflib.SequenceMatcher(None, norm(a).replace(" ", ""), norm(b).replace(" ", "")).ratio()


def is_player(name: str, tags: tuple[str, ...], thr: float = 0.8) -> bool:
    """Whether an OCR'd name is one of the user's tags (fuzzy, ignores _ . spaces)."""
    n = norm(name).replace(" ", "")
    for tag in tags:
        t = norm(tag).replace(" ", "")
        if n and t and (t in n or similar(n, t) >= thr):
            return True
    return False


# --------------------------------------------------------------------------- pixels


def hud_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of likely HUD pixels: white text, R6 blue and orange boxes."""
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    mn = np.minimum(np.minimum(b, g), r)
    mx = np.maximum(np.maximum(b, g), r)
    white = (mn > 200) & (mx - mn < 40)
    blue = (b > 150) & (r < 120) & (b - r > 80)
    orange = (r > 200) & (g > 90) & (g < 190) & (b < 100)
    return white | blue | orange


def team_color(bgr: np.ndarray) -> str | None:
    """ "ally" (blue box), "enemy" (orange box) or None for a name-box crop.

    Kill-feed boxes are semi-transparent, so the decision uses the median
    red-minus-blue of the non-text pixels (measured: orange boxes +80..+200,
    blue boxes -25..-200, grey/neutral boxes about 0).
    """
    if bgr.size == 0:
        return None
    px = bgr.reshape(-1, 3).astype(np.int16)
    bg = px[px.min(axis=1) < 170]  # drop white text pixels
    if len(bg) < 10:
        return None
    diff = float(np.median(bg[:, 2]) - np.median(bg[:, 0]))
    if diff >= 40:
        return "enemy"
    if diff <= -20:
        return "ally"
    return None


def text_hue(bgr: np.ndarray) -> str | None:
    """Dominant colour name of the text in a box crop ("white" or None if unclear).

    Game-agnostic: looks at bright pixels only (text, not the darker outline
    or the scene behind it) and names the median hue of the saturated ones.
    """
    if bgr.size == 0:
        return None
    import cv2  # noqa: PLC0415

    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    bright = hsv[hsv[:, 2] >= 150]
    if len(bright) < 8:
        return None
    sat = bright[bright[:, 1] >= 90]
    if len(sat) < max(6, 0.25 * len(bright)):
        return "white" if np.median(bright[:, 1]) < 60 else None
    h = float(np.median(sat[:, 0])) * 2  # OpenCV hue is 0-180
    for limit, name in (
        (15, "red"),
        (38, "orange"),
        (68, "yellow"),
        (160, "green"),
        (200, "cyan"),
        (250, "blue"),
        (320, "purple"),
        (361, "red"),
    ):
        if h < limit:
            return name
    return None


def colour_runs(img: np.ndarray, row: list[tuple]) -> list[tuple[str | None, str]]:
    """Split an OCR row into ``(hue, text)`` runs of equally coloured characters.

    Characters are mapped to x positions proportionally inside their box (OCR
    gives no per-character boxes), each character's column slice is coloured
    with :func:`text_hue`, and unclear characters inherit their neighbour's hue.
    Box gaps (icons) split runs too.
    """
    chars: list[tuple[str | None, str]] = []
    for bi, (box, text, _) in enumerate(row):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        n = max(1, len(text))
        if bi:
            chars.append(("|", " "))  # box boundary
        for i, ch in enumerate(text):
            a = x0 + int(i * (x1 - x0) / n)
            b = max(a + 1, x0 + int((i + 1) * (x1 - x0) / n))
            hue = None if ch.isspace() else text_hue(img[max(0, y0) : y1, max(0, a) : b])
            chars.append((hue, ch))
    # fill unclear characters from the left, then from the right
    last = None
    filled = []
    for hue, ch in chars:
        if hue not in (None, "|"):
            last = hue
        filled.append((hue if hue is not None else last, ch))
    nxt = None
    for i in range(len(filled) - 1, -1, -1):
        hue, ch = filled[i]
        if hue not in (None, "|"):
            nxt = hue
        elif hue is None:
            filled[i] = (nxt, ch)
    runs: list[tuple[str | None, str]] = []
    for hue, ch in filled:
        if hue == "|":
            if runs and runs[-1][1].strip():
                runs.append((None, ""))
            continue
        if runs and runs[-1][0] == hue:
            runs[-1] = (hue, runs[-1][1] + ch)
        else:
            runs.append((hue, ch))
    return [(h, t.strip()) for h, t in runs if t.strip()]


def _small(mask: np.ndarray, k: int = 4) -> np.ndarray:
    h, w = mask.shape[0] // k * k, mask.shape[1] // k * k
    return mask[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


# --------------------------------------------------------------------------- decoding


def iter_frames(
    path: Path,
    fps: float,
    rect: tuple[int, int, int, int],
    hw: bool,
    errors: list[str],
    start: float = 0.0,
    dur: float | None = None,
):
    """Yield ``(t, frame)`` for the cropped rectangle; stderr lines go to ``errors``.

    Only ``[start, start + dur]`` is decoded when given; ``t`` is source time.

    Raises:
        RuntimeError: If ffmpeg produced no frames at all (an
            :class:`~cutpoint.ffmpeg_utils.FFmpegError`; the caller may retry
            without hwaccel).
    """
    x, y, w, h = rect
    yield from decode_raw(
        path,
        size=(w, h),
        vf=f"fps={fps},crop={w}:{h}:{x}:{y}",
        pix_fmt="bgr24",
        start=start if start > 0 else None,
        duration=dur,
        stamp_fps=fps,
        hwaccel="cuda" if hw else None,
        fallback_cpu=False,  # the caller switches to the CPU for the remaining segments
        stderr_lines=errors,
    )


# --------------------------------------------------------------------------- tracking


class EventTracker:
    """De-duplicates repeated sightings into events with a start and an end."""

    def __init__(self, persist_s: float, fuzzy: float) -> None:
        self.persist_s = persist_s
        self.fuzzy = fuzzy
        self.events: list[dict[str, Any]] = []
        self._active: list[dict[str, Any]] = []
        self.touched: list[dict[str, Any]] = []  # events seen since reset_touched()

    def reset_touched(self) -> None:
        """Start collecting the events matched or created by the next sightings."""
        self.touched = []

    def keep_alive(self, events: list[dict[str, Any]], t: float) -> None:
        """Extend ``events`` to ``t`` without a new reading (their text is still
        on screen: the region's text layout did not change)."""
        for e in events:
            e["end"] = round(t, 2)

    @staticmethod
    def _same_victim(e: dict[str, Any], fields: dict[str, Any]) -> bool:
        """Same player victim (name + vehicle): one kill even if the actor's name
        was misread. AI targets (no vehicle group) never match this way."""
        if not (e.get("victim_vehicle") and fields.get("victim_vehicle")):
            return False
        return (
            similar(
                f"{e.get('victim')}{e['victim_vehicle']}",
                f"{fields.get('victim')}{fields['victim_vehicle']}",
            )
            >= 0.85
        )

    def see(
        self, t: float, etype: str, key: str, exclusive: bool = False, **fields: Any
    ) -> dict[str, Any] | None:
        """Record a sighting; returns the new event or None if it was a repeat.

        With ``exclusive`` an event matches at most one sighting per frame
        (time ``t``): two similar feed rows on screen at once are two events.
        """
        self._active = [e for e in self._active if t - e["end"] <= self.persist_s]
        for e in self._active:
            if exclusive and e.get("_seen_t") == t:
                continue
            # " #n" keys (repeated identical feed rows) only match exactly
            dup = " #" in key or " #" in e["_key"]
            if e["type"] == etype and (
                e["_key"] == key
                or (not dup and similar(e["_key"], key) >= self.fuzzy)
                or (exclusive and self._same_victim(e, fields))
            ):
                e["end"] = round(t, 2)
                e["sightings"] += 1
                e["_seen_t"] = t
                e["_times"].append(t)
                if fields.get("confidence", 0) > e.get("confidence", 0):
                    # keep a player flag found in any reading
                    flags = {
                        k: bool(e.get(k)) or bool(fields.get(k))
                        for k in ("player", "player_kill", "player_death")
                        if k in fields
                    }
                    e.update({k: v for k, v in fields.items() if k != "text"} | flags)
                self.touched.append(e)
                return None
        ev = {
            "t": round(t, 2),
            "end": round(t, 2),
            "type": etype,
            "_key": key,
            "sightings": 1,
            "_seen_t": t,
            "_times": [t],
            **fields,
        }
        self.events.append(ev)
        self._active.append(ev)
        self.touched.append(ev)
        return ev

    def drop_sightings(
        self, times: set[float], keep: Callable[[dict[str, Any]], bool] | None = None
    ) -> None:
        """Forget the sightings at ``times`` (menu frames), except those of the
        events ``keep`` accepts (feed sightings and kills).

        An event seen only then is dropped; the others start at their first
        remaining sighting and count only the remaining ones.
        """
        kept = []
        for e in self.events:
            if keep is not None and keep(e):
                kept.append(e)
                continue
            seen = [t for t in e["_times"] if t not in times]
            if seen:
                e["_times"], e["sightings"] = seen, len(seen)
                e["t"] = round(seen[0], 2)
                e["end"] = max(e["end"], e["t"])
                kept.append(e)
        ids = {id(e) for e in kept}
        self.events = kept
        self._active = [e for e in self._active if id(e) in ids]

    def finish(self, min_sightings: int = 1) -> list[dict[str, Any]]:
        """Events with internal keys removed (single sightings optionally dropped).

        Teamkill flags are (re)computed by :func:`link_teamkills`.
        """
        out = []
        for e in self.events:
            if e.get("unknown") and e["sightings"] < 2:
                continue  # one-frame OCR noise in a generic text region
            if e["sightings"] >= min_sightings or e["type"] in ("banner", "vs"):
                out.append({k: v for k, v in e.items() if not k.startswith("_")})
        return out


# --------------------------------------------------------------------------- OCR logic


OCR_DET_SIDE = 960  # the OCR layer's detection size limit (long side, px)


class OcrEngine:
    """Thin wrapper around RapidOCR (onnxruntime, CPU)."""

    def __init__(self, det_side: int = OCR_DET_SIDE) -> None:
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

        self.det_side = det_side  # part of the raw OCR cache key

        # Default det config upscales the short side to 736 px ("limit_type: min"),
        # which makes wide HUD strips ~10x slower; cap the long side instead.
        self.engine = RapidOCR(det_limit_type="max", det_limit_side_len=det_side)

    def detect(self, bgr: np.ndarray, scale: float = 1.0) -> list[list]:
        """Text boxes only (no recognition), used by the "det" gate.

        ``scale`` < 1 detects on a downscaled copy (much cheaper); boxes are
        returned in the coordinates of ``bgr``.
        """
        img = bgr
        if scale < 1.0:
            import cv2  # noqa: PLC0415

            img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result, _ = self.engine(np.ascontiguousarray(img), use_cls=False, use_rec=False)
        boxes = [b.tolist() if hasattr(b, "tolist") else b for b in (result or [])]
        if scale < 1.0:
            boxes = [[[p[0] / scale, p[1] / scale] for p in b] for b in boxes]
        return boxes

    def read_rows(
        self,
        bgr: np.ndarray,
        rows: list[tuple[float, float, float, float]],
        min_score: float,
        pad: float = 6.0,
    ) -> list[tuple[list, str, float]]:
        """Recognise text rows ``(y0, x0, y1, x1)`` of ``bgr`` without a second
        detection pass (one recognition per row strip, ~25-60 ms each).

        Returns items shaped like :meth:`__call__` (box = padded row rectangle).
        """
        h, w = bgr.shape[:2]
        out = []
        for y0, x0, y1, x1 in rows:
            a, b = int(max(0, y0 - pad / 2)), int(min(h, y1 + pad / 2))
            c, d = int(max(0, x0 - 2 * pad)), int(min(w, x1 + pad))
            if b - a < 6 or d - c < 6:
                continue
            res, _ = self.engine(
                np.ascontiguousarray(bgr[a:b, c:d]), use_det=False, use_cls=False, use_rec=True
            )
            for item in res or []:
                text, score = str(item[0]), float(item[1])
                if score >= min_score and text.strip():
                    out.append(([[c, a], [d, a], [d, b], [c, b]], text, score))
        return out

    def __call__(self, bgr: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        result, _ = self.engine(np.ascontiguousarray(bgr), use_cls=False)
        return [
            (box, text, float(score))
            for box, text, score in (result or [])
            if float(score) >= min_score and text.strip()
        ]


def _rows(items: list[tuple[list, str, float]], tol: float) -> list[list[tuple]]:
    """Group OCR boxes into text rows by vertical centre, each row sorted by x."""
    rows: list[list[tuple]] = []
    for it in sorted(items, key=lambda i: np.mean([p[1] for p in i[0]])):
        cy = np.mean([p[1] for p in it[0]])
        if rows and abs(np.mean([p[1] for p in rows[-1][0][0]]) - cy) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return [sorted(r, key=lambda i: min(p[0] for p in i[0])) for r in rows]


def _merge_adjacent(items: list[tuple], gap: float) -> list[tuple]:
    """Join OCR boxes of one row that touch (no weapon icon between them).

    OCR sometimes splits one name ("layer" + "ame" for PlayerName); the
    killer and victim boxes are always separated by the weapon icon (~40 px).
    """
    out: list[tuple] = []
    for it in items:
        if out:
            prev = out[-1]
            prev_right = max(p[0] for p in prev[0])
            left = min(p[0] for p in it[0])
            if left - prev_right < gap:
                box = [
                    [min(prev[0][0][0], it[0][0][0]), prev[0][0][1]],
                    [max(prev_right, max(p[0] for p in it[0])), prev[0][1][1]],
                    [max(prev_right, max(p[0] for p in it[0])), prev[0][2][1]],
                    [min(prev[0][3][0], it[0][3][0]), prev[0][3][1]],
                ]
                out[-1] = (box, prev[1] + it[1], min(prev[2], it[2]))
                continue
        out.append(it)
    return out


def _box_crop(img: np.ndarray, box: list) -> np.ndarray:
    xs, ys = [int(p[0]) for p in box], [int(p[1]) for p in box]
    return img[max(0, min(ys)) : max(ys), max(0, min(xs)) : max(xs)]


class Interpreter:
    """Turns OCR results of one region into tracked events."""

    def __init__(self, params: OcrParams, scale: float, preset: HudPreset | None = None) -> None:
        self.params = params
        self.scale = scale
        self.preset = preset
        self.generic = preset is not None and not preset.builtin
        self.phrases = preset.phrases if preset else ()
        self.tags = tuple(
            dict.fromkeys([*params.player_tags, *(preset.player_tags if preset else ())])
        )
        self.tracker = EventTracker(params.persist_s, params.fuzzy)
        self.last_vs: tuple[int, int] | None = None
        self.feed = FeedParser(preset.feed if preset else {}, self.tags, scale)

    def _phrases(self, t: float, region: Region, text: str, conf: float) -> bool:
        found = False
        for ph in self.phrases:
            if fuzzy_contains(text, ph, self.params.fuzzy):
                # Keep trailing digits for "WON ROUND 5"-style banners.
                m = re.search(r"ROUND\s*(\d{1,2})\b", norm(text)) if "ROUND" in ph.upper() else None
                label = f"{ph} {m.group(1)}" if m else ph
                self.tracker.see(
                    t,
                    "banner",
                    ph,
                    text=label,
                    raw=text[:120],
                    confidence=round(conf, 3),
                    region=region.name,
                )
                found = True
        return found

    def handle(
        self, t: float, region: Region, img: np.ndarray, items: list[tuple[list, str, float]]
    ) -> None:
        """Interpret OCR ``items`` of ``region`` (crop ``img``) at time ``t``."""
        if not items:
            return
        if self.generic:
            self._handle_generic(t, region, img, items)
            return
        tags = self.params.player_tags
        full = " ".join(i[1] for i in items)
        conf = float(np.mean([i[2] for i in items]))
        for m in _POINTS_RE.finditer(full):
            self.tracker.see(
                t,
                "points",
                m.group(0),
                text=f"+{m.group(1)}",
                confidence=round(conf, 3),
                region=region.name,
            )
        if region.kind == "counter":
            m = _VS_RE.search(full.upper().replace("O", "0"))
            if m:
                pair = (int(m.group(1)), int(m.group(2)))
                if pair != self.last_vs and pair[0] <= 5 and pair[1] <= 5:
                    self.last_vs = pair
                    self.tracker.see(
                        t,
                        "vs",
                        f"{pair[0]}v{pair[1]}",
                        text=f"{pair[0]} VS {pair[1]}",
                        confidence=round(conf, 3),
                        region=region.name,
                    )
            return
        if region.kind in ("banner", "notice"):
            self._phrases(t, region, full, conf)
            return
        if region.kind == "log":
            for row in _rows(items, 10 * self.scale):
                line = " ".join(i[1] for i in row)
                c = float(np.mean([i[2] for i in row]))
                low = line.lower()
                if any(
                    k in low
                    for k in ("destroyed", "shot down", "set afire", "has achieved", "critical")
                ):
                    self.tracker.see(
                        t,
                        "log",
                        norm(line),
                        text=line[:160],
                        confidence=round(c, 3),
                        region=region.name,
                        player=is_player(line, tags),
                    )
            return
        # killfeed
        for row in _rows(items, 12 * self.scale):
            line = " ".join(i[1] for i in row)
            c = float(np.mean([i[2] for i in row]))
            if self._phrases(t, region, line, c):
                continue
            names = _merge_adjacent([i for i in row if len(norm(i[1])) >= 2], gap=14 * self.scale)
            if len(names) < 2:
                continue
            killer, victim = names[0], names[-1]
            k_team = team_color(_box_crop(img, killer[0]))
            v_team = team_color(_box_crop(img, victim[0]))
            if k_team is None and v_team is None:
                continue  # not a kill-feed row (no coloured name boxes)
            teamkill = k_team == "ally" and v_team == "ally"
            player = is_player(killer[1], tags) or is_player(victim[1], tags)
            text = f"{killer[1]} > {victim[1]}"
            self.tracker.see(
                t,
                "kill",
                norm(text),
                text=text,
                killer=killer[1],
                victim=victim[1],
                killer_team=k_team,
                victim_team=v_team,
                teamkill=teamkill,
                player=player,
                player_kill=is_player(killer[1], tags),
                player_death=is_player(victim[1], tags),
                confidence=round(c, 3),
                region=region.name,
            )

    # ------------------------------------------------------------ generic presets
    def _generic_phrases(self, t: float, region: Region, text: str, conf: float) -> bool:
        """Phrase matches, keeping only the longest of nested matches
        ("YOU HAVE BEEN DESTROYED", not also "DESTROYED"); the text after the
        phrase is kept as ``detail`` ("by -BRoFS- Sinistro9121 (Strv m/40L)")."""
        hits = [ph for ph in self.phrases if fuzzy_contains(text, ph, self.params.fuzzy)]
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

    def _handle_generic(
        self, t: float, region: Region, img: np.ndarray, items: list[tuple[list, str, float]]
    ) -> None:
        """JSON-preset interpretation: feeds, banners, notices and big text."""
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
            if self._generic_phrases(t, region, line, c):
                continue
            if region.kind == "bigtext" and h < 1.4 * min_h:
                continue  # unknown text must be really big to count as a banner
            letters = sum(ch.isalpha() for ch in line)
            if letters < 4 or letters < 0.6 * len(line.replace(" ", "")):
                continue  # numbers, ranges, reticle marks
            etype = "notice" if region.kind == "notice" else "bigtext"
            self.tracker.see(
                t,
                etype,
                norm(line),
                text=line[:120],
                confidence=round(c, 3),
                region=region.name,
                text_h=round(h / self.scale),
                unknown=True,
            )


_CLAN_RE = re.compile(r"^\s*(\[[^\]]{1,8}\]|\^[^\^]{1,8}\^|=[^=]{1,8}=|-[^-]{1,8}-)\s*")
DEFAULT_KILL_WORDS = {
    "destroyed": "kill",
    "shot down": "kill",
    "killed": "kill",
    "eliminated": "kill",
    "wrecked": "kill",
    "set afire": "fire",
    "critically damaged": "crit",
    "severely damaged": "damage",
    "assist": "assist",
    "has crashed": "crash",
    "crashed": "crash",
    "has been wrecked": "crash",
}


def paren_groups(text: str) -> list[tuple[int, int]]:
    """Top-level ``(...)`` spans (OCR full-width parens accepted; an unclosed
    group at the end of the text runs to the end)."""
    out, depth, start = [], 0, -1
    for i, ch in enumerate(text):
        if ch in "(\uff08":
            if depth == 0:
                start = i
            depth += 1
        elif ch in ")\uff09" and depth:
            depth -= 1
            if depth == 0:
                out.append((start, i + 1))
    if depth and start >= 0:
        out.append((start, len(text)))
    return out


def _alnum_len(text: str) -> int:
    """Letters and digits in any script (Korean names count too)."""
    return sum(ch.isalnum() for ch in text)


def _before_group(text: str) -> str:
    """Text before the first ``(`` group (the whole text if there is none)."""
    g = paren_groups(text)
    return text[: g[0][0]] if g else text


def _last_token(text: str) -> str:
    """Name right before a vehicle group: the last token, clan tags dropped."""
    toks = [t for t in re.split(r"\s+", text.strip()) if t]
    while toks and _CLAN_RE.fullmatch(toks[-1] + " "):
        toks.pop()
    return _CLAN_RE.sub("", toks[-1]) if toks else ""


def split_name(text: str) -> tuple[str, str | None]:
    """``"[CLAN] Name (Vehicle)"`` -> ``("Name", "Vehicle")`` (tags stripped)."""
    t = _CLAN_RE.sub("", text.strip())
    groups = paren_groups(t)
    if groups and groups[0][0] > 0:
        a, b = groups[0]
        name = _CLAN_RE.sub("", t[:a].strip()).strip() or _last_token(t[:a])
        return re.sub(r"^[\W_]+", "", name).rstrip(" -_>"), t[a + 1 : b].strip(" )\uff09") or None
    return t.strip(" -_>"), None


KILL_WORD_GLUE = 7  # kill words this long may be joined to a name on one side


def _is_bound(text: str, i: int) -> bool:
    """Whether position ``i`` of ``text`` is a word boundary, as in
    :func:`fuzzy_contains`: an end of the text, a character that is no letter
    or digit on either side, a camelCase step ("yA") or a letter/digit step."""
    if i <= 0 or i >= len(text):
        return True
    a, b = text[i - 1], text[i]
    if not (a.isalnum() and b.isalnum()):
        return True
    return (a.islower() and b.isupper()) or a.isdigit() != b.isdigit()


class FeedParser:
    """Game-agnostic kill-feed parsing for "feed" regions.

    Two forms are recognised (both from one region crop):

    * a row containing a kill word ("A destroyed B", "A shot down B", "A set
      afire B"): actor before the word, target after it;
    * icon feeds without words, where an entry is a coloured *actor* line and
      a differently coloured *target* line (``"entry": "two_line"``, War
      Thunder) or two coloured names on one row (``"entry": "one_line"``).

    Team colours come from the preset (``"colors": {"red": "enemy", "blue":
    "ally", ...}``, keyed by the colour names :func:`text_hue` returns). A kill
    is the player's when the actor matches one of the player tags (fuzzy,
    ignoring ``_ . -``).
    """

    def __init__(self, cfg: dict[str, Any], tags: tuple[str, ...], scale: float) -> None:
        self.words = {
            k.upper(): v for k, v in (cfg.get("kill_words") or DEFAULT_KILL_WORDS).items()
        }
        self.colors: dict[str, str] = dict(cfg.get("colors") or {})
        self.entry = cfg.get("entry", "auto")
        self.tags = tags
        self.scale = scale
        self.row_tol = float(cfg.get("row_tol", 10)) * scale
        self.min_name = int(cfg.get("min_name", 3))
        self._img: np.ndarray | None = None

    def _team(self, img: np.ndarray, row: list[tuple]) -> str | None:
        hues = [text_hue(_box_crop(img, i[0])) for i in row]
        hues = [h for h in hues if h]
        if not hues:
            return None
        hue = max(set(hues), key=hues.count)
        return self.colors.get(hue, hue)

    def _event(
        self,
        actor: str,
        target: str,
        verb: str,
        a_team: str | None,
        t_team: str | None,
        conf: float,
        raw: str,
    ) -> dict[str, Any] | None:
        a_name, a_veh = split_name(actor)
        t_name, t_veh = split_name(target)
        if _alnum_len(t_name) < self.min_name:
            return None
        if verb == "kill" and _alnum_len(a_name) < self.min_name:
            return None
        teamkill = bool(a_team and t_team and a_team == t_team == "ally" and verb == "kill")
        # The player check uses the whole text before the vehicle group: OCR
        # splits names ("Player Name") and glues clan tags to them.
        p_actor = bool(actor.strip()) and is_player(_before_group(actor), self.tags)
        p_target = is_player(_before_group(target), self.tags)
        etype = "kill" if verb == "kill" else verb
        text = (
            f"{a_name}{f' ({a_veh})' if a_veh else ''} > {t_name}{f' ({t_veh})' if t_veh else ''}"
            if verb == "kill"
            else raw[:120]
        )
        # key: actor + vehicle + verb. The target's OCR varies too much ("Churchill
        # VIl" / "M61 shot Churchill Vll" when the colour split fails); repeated
        # kills by one actor are kept apart by exclusive per-frame matching.
        key = norm(f"{a_name} {a_veh or ''}:{verb}") if a_name else norm(f"{t_name}:{verb}")
        return {
            "type": etype,
            "key": key,
            "text": text,
            "killer": a_name,
            "victim": t_name,
            "killer_vehicle": a_veh,
            "victim_vehicle": t_veh,
            "killer_team": a_team,
            "victim_team": t_team,
            "teamkill": teamkill,
            "player": p_actor or p_target,
            "player_kill": p_actor and verb == "kill",
            "player_death": p_target and verb in ("kill", "crash"),
            "verb": verb,
            "raw": raw[:160],
            "confidence": round(conf, 3),
        }

    def _word_split(self, line: str) -> tuple[str, str, str] | None:
        """(actor, target, verb) around the longest kill word of ``line``, or None.

        A kill word matches whole words only (case ignored): a word of fewer
        than :data:`KILL_WORD_GLUE` letters starts and ends at a word boundary of
        :func:`_is_bound`, so "assist" is not found in "assistance" or "killed"
        in "skilled" but is in "PlayerAssistFoe". A longer one needs a boundary
        on one side only, since OCR often joins a name to it ("Ashdestroyed
        Kip", "Kiphas crashed").
        """
        for word, verb in sorted(self.words.items(), key=lambda kv: -len(kv[0])):
            pat = r"\s+".join(map(re.escape, word.split()))
            glue = len(word.replace(" ", "")) >= KILL_WORD_GLUE
            for m in re.finditer(pat, line, re.IGNORECASE):
                a, b = _is_bound(line, m.start()), _is_bound(line, m.end())
                if (a or b) if glue else (a and b):
                    return line[: m.start()].strip(), line[m.end() :].strip(), verb
        return None

    def _colour_target(
        self, row: list[tuple] | None, trailing: str
    ) -> tuple[str | None, str | None]:
        """(target, weapon) from the colour runs of a row's trailing text."""
        if self._img is None or not row:
            return None, None
        runs = colour_runs(self._img, row)
        if len(runs) < 2:
            return None, None
        hue, text = runs[-1]
        prev = runs[-2][0]
        if hue in (None, "white") or hue == prev or _alnum_len(text) < 5:
            return None, None
        if norm(text).replace(" ", "") not in norm(trailing).replace(" ", ""):
            return None, None
        weapon = trailing[: max(0, len(trailing) - len(text))].strip(" -=_>") or None
        return text, weapon

    def _parse_paren(self, lines: list[tuple], used: set[int]) -> list[dict[str, Any]]:
        """Icon feeds whose names carry the vehicle in parentheses.

        * one row, two groups: ``Killer (Veh) <weapon> Victim (Veh)``;
        * wrapped entry: an actor row ``Killer (Veh) <weapon>`` (one group plus
          trailing text) followed by a victim row ``Victim (Veh)`` (one group
          at the end).

        Who did what is decided by position, not colour: the player's tag
        before the first group means the player is the actor (a player kill),
        after it the player is the victim. Team colours are recorded when the
        text hue maps to a team in the preset's ``colors``.
        """
        out: list[dict[str, Any]] = []
        k = 0
        while k < len(lines):
            if k in used:
                k += 1
                continue
            line, conf, team, _ = lines[k]
            groups = paren_groups(line)
            victim_next = False
            if k + 1 < len(lines) and k + 1 not in used:
                ng = paren_groups(lines[k + 1][0])
                victim_next = len(ng) == 1 and not lines[k + 1][0][ng[0][1] :].strip(" -=_>)\uff09")
            if len(groups) >= 2 and victim_next and len(line[groups[0][1] :].strip(" -=_>")) >= 2:
                # a victim row never stands alone: this row is the actor row and
                # its weapon contains parentheses ("BR-350A (MD-5 fuze)")
                groups = groups[:1]
            if len(groups) >= 2:
                (_, a1), (b0, b1) = groups[0], groups[-1]
                actor = line[:a1]
                victim_name = _last_token(line[a1:b0])
                weapon = line[a1:b0][: max(0, len(line[a1:b0]) - len(victim_name))]
                ev = self._event(
                    actor, f"{victim_name} {line[b0:b1]}", "kill", team, None, conf, line
                )
                if ev:
                    ev["weapon"] = weapon.strip(" -=_>") or None
                    out.append(ev)
                k += 1
                continue
            trailing = line[groups[0][1] :].strip(" -=_>") if groups else ""
            if len(groups) == 1 and len(trailing) >= 2 and not victim_next:
                # single row, target without a vehicle group (AI vehicle such as
                # "Churchill VII"): the target is the last colour run
                target, weapon = self._colour_target(lines[k][3], trailing)
                split = target is not None
                if not split:
                    target, weapon = trailing, None
                if target:
                    ev = self._event(line[: groups[0][1]], target, "kill", team, None, conf, line)
                    if ev:
                        ev["weapon"] = weapon
                        ev["ai_target"] = True
                        ev["target_split"] = split
                        out.append(ev)
                k += 1
                continue
            if len(groups) == 1 and len(trailing) >= 2 and k + 1 < len(lines) and k + 1 not in used:
                nxt, nconf, nteam, _ = lines[k + 1]
                ng = paren_groups(nxt)
                if len(ng) == 1 and not nxt[ng[0][1] :].strip(" -=_>)\uff09"):
                    ev = self._event(
                        line[: groups[0][1]],
                        nxt,
                        "kill",
                        team,
                        nteam,
                        min(conf, nconf),
                        f"{line} | {nxt}",
                    )
                    if ev:
                        ev["weapon"] = trailing or None
                        out.append(ev)
                    k += 2
                    continue
            k += 1
        return out

    def parse(self, img: np.ndarray, items: list[tuple[list, str, float]]) -> list[dict[str, Any]]:
        """Events (dicts with ``type`` and ``key``) for one OCR'd feed crop."""
        self._img = img
        rows = _rows(items, self.row_tol)
        lines = [
            (
                " ".join(i[1] for i in r).strip(),
                float(np.mean([i[2] for i in r])),
                self._team(img, r),
                r,
            )
            for r in rows
        ]
        out: list[dict[str, Any]] = []
        used: set[int] = set()
        for k, (line, conf, team, _row) in enumerate(lines):
            sp = self._word_split(line)
            if sp is None:
                continue
            actor, target, verb = sp
            if verb in ("crash",):
                actor, target = "", actor
            ev = self._event(actor, target, verb, team, None, conf, line)
            if ev:
                out.append(ev)
                used.add(k)
        if self.entry == "paren":
            out += self._parse_paren(lines, used)
            return out
        if self.entry in ("two_line", "auto"):
            k = 0
            while k < len(lines) - 1:
                if k in used or k + 1 in used:
                    k += 1
                    continue
                (a, ca, ta, _), (b, cb, tb, _) = lines[k], lines[k + 1]
                teams = {"ally", "enemy", "squad", "self"}
                if (
                    ta in teams
                    and tb in teams
                    and ta != tb
                    and {ta, tb} != {"ally", "squad"}
                    and {ta, tb} != {"ally", "self"}
                    and {ta, tb} != {"squad", "self"}
                ):
                    # actor line may end with the weapon after the icon: keep the name
                    ev = self._event(a, b, "kill", ta, tb, min(ca, cb), f"{a} | {b}")
                    if ev:
                        out.append(ev)
                        used |= {k, k + 1}
                        k += 2
                        continue
                k += 1
        if self.entry in ("one_line", "auto"):
            for k, (line, conf, _, row) in enumerate(lines):
                if k in used:
                    continue
                names = _merge_adjacent(
                    [i for i in row if len(norm(i[1])) >= 2], gap=14 * self.scale
                )
                if len(names) < 2:
                    continue
                ka = self._team(img, [names[0]])
                kb = self._team(img, [names[-1]])
                if ka and kb and ka != kb and {ka, kb} <= {"ally", "enemy", "squad", "self"}:
                    ev = self._event(names[0][1], names[-1][1], "kill", ka, kb, conf, line)
                    if ev:
                        out.append(ev)
        return out


def group_rows(
    boxes: list[list], min_h: float, tol: float
) -> list[tuple[float, float, float, float, float]]:
    """Group detected text boxes into rows ``(cy, x0, x1, y0, y1)``, top to bottom.

    Boxes shorter than ``min_h`` are ignored; boxes whose vertical centres are
    within ``tol`` of the previous row's centre join that row.
    """
    rows: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: sum(p[1] for p in b)):
        y0, y1 = min(p[1] for p in b), max(p[1] for p in b)
        if min_h and y1 - y0 < min_h:
            continue
        cy, x0, x1 = (y0 + y1) / 2, min(p[0] for p in b), max(p[0] for p in b)
        if rows and abs(rows[-1][0] - cy) <= tol:
            r = rows[-1]
            r[1], r[2], r[3], r[4] = min(r[1], x0), max(r[2], x1), min(r[3], y0), max(r[4], y1)
        else:
            rows.append([cy, x0, x1, y0, y1])
    return [tuple(r) for r in rows]


def keep_segments(duration: float, avoid: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``[0, duration]`` minus the ``avoid`` ranges (each widened by 2 s)."""
    segs = [(0.0, duration)]
    for a, b in sorted(avoid):
        a, b = max(0.0, a - 2.0), b + 2.0
        nxt = []
        for s0, s1 in segs:
            if b <= s0 or a >= s1:
                nxt.append((s0, s1))
                continue
            if a > s0:
                nxt.append((s0, a))
            if b < s1:
                nxt.append((b, s1))
        segs = nxt
    return [(a, b) for a, b in segs if b - a >= 1.0]


def link_teamkills(
    events: list[dict[str, Any]],
    window_s: float = 3.0,
    tags: tuple[str, ...] = DEFAULT_PLAYER_TAGS,
    generic_teamkills: bool = False,
) -> None:
    """Post-process kill events in place (also applied to cached results).

    * Drops "kills" whose killer + victim text is one of the player's tags:
      OCR split a single name into two boxes ("layer" > "ame"). Built-in
      (R6) events only; generic feed events carry ``verb`` and keep player
      deaths.
    * Generic feed events: ``teamkill`` only when ``generic_teamkills`` (the
      preset's ``feed.teamkills``) and both name colours read as ally.
    * (Re)computes ``teamkill``.

    A kill is a teamkill when both name boxes are blue, or when a "REVERSE
    FRIENDLY FIRE" banner shows within ``window_s`` and neither name box is
    orange (the victim's box is often grey, so colour alone misses it).
    Applied to cached results too, so the rule can change without re-OCR.
    """
    events[:] = [
        e
        for e in events
        if e.get("verb")
        or not (
            e.get("type") == "kill"
            and not is_player(str(e.get("killer", "")), tags)
            and is_player(f"{e.get('killer', '')}{e.get('victim', '')}", tags)
        )
    ]
    rff = [
        e["t"]
        for e in events
        if e.get("type") == "banner" and str(e.get("text", "")).startswith("REVERSE FRIENDLY FIRE")
    ]
    for e in events:
        if e.get("type") != "kill":
            continue
        if e.get("verb"):
            # generic feeds (JSON presets): colour teamkills only if the preset
            # trusts its colours (``"feed": {"teamkills": true}``)
            e["teamkill"] = bool(
                generic_teamkills
                and e.get("killer_team") == "ally"
                and e.get("victim_team") == "ally"
            )
            continue
        k, v = e.get("killer_team"), e.get("victim_team")
        near = any(abs(e["t"] - t) <= window_s for t in rff)
        e["teamkill"] = (k == "ally" and v == "ally") or (near and "enemy" not in (k, v))


def derive_streaks(events: list[dict[str, Any]], window_s: float = 20.0) -> None:
    """Add ``multikill`` events for 2+ player kills within ``window_s`` (in place).

    Only generic-feed kills (JSON presets, they carry ``verb``) are considered,
    so built-in R6 results are unchanged; a self-kill (``player_death`` set too)
    is no kill here. Idempotent: old derived events are dropped first.
    """
    events[:] = [e for e in events if e.get("type") != "multikill"]
    kills = sorted(
        (
            e
            for e in events
            if e.get("type") == "kill"
            and e.get("verb")
            and e.get("player_kill")
            and not e.get("player_death")
        ),
        key=lambda e: e["t"],
    )
    i = 0
    while i < len(kills):
        j = i
        while j + 1 < len(kills) and kills[j + 1]["t"] - kills[j]["t"] <= window_s:
            j += 1
        if j > i:
            n = j - i + 1
            word = {2: "DOUBLE", 3: "TRIPLE"}.get(n, f"{n}x")
            events.append(
                {
                    "t": kills[i]["t"],
                    "end": kills[j]["t"],
                    "type": "multikill",
                    "text": f"{word} KILL ({', '.join(k['victim'] for k in kills[i : j + 1])})",
                    "kills": n,
                    "player": True,
                    "confidence": 1.0,
                    "region": kills[i].get("region"),
                    "kind": "ocr",
                    "derived": True,
                }
            )
        i = j + 1
    events.sort(key=lambda e: e["t"])


LIST_MENU_COVER = 0.25  # centre cover of a list menu (2 columns: ~0.33) when the jump is off

# Static screens (rule "static"): hangar, crew and loadout menus show as little text as a busy
# HUD (6-17 boxes, centre cover 0.04-0.12), so the box rules miss them, but they hold still
# while gameplay moves. Every OCR frame takes a STATIC_THUMB (w, h) grayscale thumbnail of the
# HUD area it already decoded (:func:`_thumb_bgr`, :func:`_grey`; no extra decode) and :class:`StillRuns` gives it
# the id of its still run (``"still"`` in the frame record; no thumbnail is kept). A run goes
# on while thumbnails stay within STATIC_MAX_DIFF of its first frame (mean absolute difference
# in 0-255 grey levels: still video and its compression noise stay below 1, a moving game view
# changes by tens of levels, a menu cursor or a small animation by a few) and no frame is
# missing (a gap over STATIC_MAX_GAP_S, e.g. a skipped bad range, ends it). The frames within
# STATIC_FEED_GUARD_S of feed activity (a feed sighting or a kill) are taken out, so kills in
# still gameplay (scoped, spectating) are never dropped; the rest of a run is a static screen
# where at least STATIC_MIN_FRAMES frames hold still for at least STATIC_MIN_S when it shows new
# menu text, else STATIC_BARE_MIN_S. New menu text is at least STATIC_TEXT_LINES texts the
# generic interpreter reads that are no preset phrase (``unknown`` notice or big text lines, so
# not numbers or reticle marks), first read in the part or up to STATIC_FEED_GUARD_S before it
# (a menu shows its words when it opens) and each on screen for at least STATIC_TEXT_SHARE of
# the part: a menu shows many lines that stay, a gameplay notice is one line or one box that
# flashes, so a notice read in still gameplay never puts it on the STATIC_MIN_S path.
# Still gameplay (scoped, binoculars) holds still for about 6 s, a result screen 6.7 s and menus
# 17-56 s on the HUD area, so the 5 s minimum alone has no margin; a phrase banner in still
# gameplay is no menu text, so it never supports dropping itself (menus show phrases too, so
# phrases cannot guard either). The test runs only when the decoded HUD area is at least
# STATIC_MIN_AREA of the frame: a few small corner regions see too little of the screen (still
# sky or ground would pass as a menu).
STATIC_THUMB = (32, 18)
STATIC_MAX_DIFF = 3.0
STATIC_MIN_S = 5.0
STATIC_BARE_MIN_S = 10.0
STATIC_MIN_FRAMES = 3
STATIC_FEED_GUARD_S = 2.0
STATIC_MAX_GAP_S = 1.0  # at least; 2 frame intervals at low OCR frame rates
STATIC_MIN_AREA = 0.25
STATIC_TEXT_LINES = 2
STATIC_TEXT_SHARE = 0.5


def _thumb_bgr(frame: np.ndarray) -> np.ndarray:
    """The :data:`STATIC_THUMB` BGR thumbnail (uint8) of a decoded BGR frame, as the raw
    OCR cache keeps it."""
    import cv2  # noqa: PLC0415

    return cv2.resize(frame, STATIC_THUMB, interpolation=cv2.INTER_AREA)


def _grey(thumb: np.ndarray) -> np.ndarray:
    """A :func:`_thumb_bgr` thumbnail as the grayscale (float32) :class:`StillRuns` takes."""
    return thumb.astype(np.float32).mean(axis=2)


class StillRuns:
    """Still run ids of thumbnails in time order, computed online: only the run's
    first thumbnail and the last time are kept (see :data:`STATIC_THUMB`)."""

    def __init__(self, fps: float) -> None:
        self.max_gap = max(STATIC_MAX_GAP_S, 2 / fps) if fps > 0 else STATIC_MAX_GAP_S
        self.anchor: np.ndarray | None = None
        self.last_t: float | None = None
        self.run = -1

    def add(self, t: float, thumb: np.ndarray) -> int:
        """The run id of the frame at ``t`` with ``thumb``."""
        if (
            self.anchor is None
            or self.last_t is None
            or t - self.last_t > self.max_gap
            or float(np.mean(np.abs(thumb - self.anchor))) > STATIC_MAX_DIFF
        ):
            self.run += 1
            self.anchor = thumb
        self.last_t = t
        return self.run


def still_stretches(frames: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """``frames`` (in time order) cut into their still runs (``"still"`` ids from
    :class:`StillRuns`). A frame without one is a run of its own (records made
    before still runs existed, or a HUD area too small for the test)."""
    runs: list[list[dict[str, Any]]] = []
    prev: Any = None
    for f in frames:
        run = f.get("still")
        if run is None or run != prev or not runs:
            runs.append([f])
        else:
            runs[-1].append(f)
        prev = run
    return runs


def static_parts(
    frames: list[dict[str, Any]],
    feed_times: set[float],
    text_spans: list[tuple[float, float]],
) -> list[tuple[list[dict[str, Any]], bool]]:
    """The parts of still runs that may be static screens (see :data:`STATIC_THUMB`):
    frames of a still run further than :data:`STATIC_FEED_GUARD_S` from every time
    of ``feed_times`` (feed sightings and kills), each with whether it shows new menu
    text: at least :data:`STATIC_TEXT_LINES` of the ``text_spans`` (first read, end)
    first read in the part or up to :data:`STATIC_FEED_GUARD_S` before it, each
    overlapping at least :data:`STATIC_TEXT_SHARE` of the part."""
    busy, spans = sorted(feed_times), sorted(text_spans)
    starts = [s for s, _ in spans]

    def any_in(times: list[float], a: float, b: float) -> bool:
        i = bisect.bisect_left(times, a)
        return i < len(times) and times[i] <= b

    def menu_text(a: float, b: float) -> bool:
        lo = bisect.bisect_left(starts, a - STATIC_FEED_GUARD_S)
        hi = bisect.bisect_right(starts, b)
        stay = sum(min(e, b) - max(s, a) >= STATIC_TEXT_SHARE * (b - a) for s, e in spans[lo:hi])
        return stay >= STATIC_TEXT_LINES

    out: list[tuple[list[dict[str, Any]], bool]] = []
    for run in still_stretches(frames):
        part: list[dict[str, Any]] = []
        for f in [*run, None]:  # None closes the last part
            if (
                f is not None
                and f.get("still") is not None
                and not any_in(busy, f["t"] - STATIC_FEED_GUARD_S, f["t"] + STATIC_FEED_GUARD_S)
            ):
                part.append(f)
                continue
            if part:
                a, b = part[0]["t"], part[-1]["t"]
                out.append((part, menu_text(a, b)))
            part = []
    return out


def static_frames(
    frames: list[dict[str, Any]],
    feed_times: set[float],
    text_spans: list[tuple[float, float]] | None = None,
) -> set[Any]:
    """Ids of the frames of a static screen: a part of :func:`static_parts` with at
    least :data:`STATIC_MIN_FRAMES` frames that holds still for at least
    :data:`STATIC_MIN_S` when it shows new menu text (``text_spans``), else
    :data:`STATIC_BARE_MIN_S`."""
    out: set[Any] = set()
    for part, text in static_parts(frames, feed_times, text_spans or []):
        need = STATIC_MIN_S if text else STATIC_BARE_MIN_S
        if len(part) >= STATIC_MIN_FRAMES and part[-1]["t"] - part[0]["t"] >= need:
            out.update(p["id"] for p in part)
    return out


def menu_times(
    frames: list[dict[str, Any]],
    params: DiscoverParams | None = None,
    feed_times: set[float] | None = None,
) -> set[float]:
    """Times of the menu and scoreboard frames among ``frames`` of one source.

    ``frames`` are ``{"id", "t", "source", "boxes"}`` records of the text boxes
    the OCR layer detected per frame (reference pixels); the test is
    discovery's :func:`~cutpoint.events.discover.gameplay_frames` (default
    params), with the source's median box count taken over these frames. The
    region boxes of a frame are few (a median of 0 to 6), so the box-count
    jump applies only when the median is at least ``menu_min_boxes /
    menu_jump``. Below that, a frame with ``menu_min_boxes`` boxes is a menu
    when they cover the screen centre (``menu_cover``) or at least
    :data:`LIST_MENU_COVER` of it (a 1- or 2-column list or scoreboard).
    With ``feed_times`` (the times of feed sightings and kills), frames with a
    ``still`` run id that hold still are menus too (:func:`static_frames`).
    """
    rules, _ = menu_rules(frames, params, feed_times)
    return {f["t"] for f in frames if rules[f["id"]] != "none"}


def menu_rules(
    frames: list[dict[str, Any]],
    params: DiscoverParams | None = None,
    feed_times: set[float] | None = None,
    text_spans: list[tuple[float, float]] | None = None,
) -> tuple[dict[Any, str], dict[str, float]]:
    """The :func:`menu_times` decision per frame id: the rule that made it a menu
    (``cover``, ``jump``, ``list`` or ``static``) or ``none``, and the source's
    numbers (``median`` box count, ``min_boxes``, the ``cover``, ``list`` and
    ``jump`` thresholds, inf when off, and the ``static_s`` (with new menu text,
    ``text_spans``), ``static_bare_s`` (without) and ``static_diff`` of the static
    rule, which runs only with ``feed_times``)."""
    params = params or DiscoverParams()
    median = float(np.median([len(f["boxes"]) for f in frames])) if frames else 0.0
    lists = median < params.menu_min_boxes / params.menu_jump
    if lists:
        params = replace(params, menu_jump=float("inf"))
    keep = {f["id"] for f in gameplay_frames(frames, params)}
    static = static_frames(frames, feed_times, text_spans) if feed_times is not None else set()
    rules = {}
    for f in frames:
        if f["id"] not in keep:
            rules[f["id"]] = "cover" if _centre_cover(f) >= params.menu_cover else "jump"
        elif (
            lists
            and len(f["boxes"]) >= params.menu_min_boxes
            and _centre_cover(f) >= LIST_MENU_COVER
        ):
            rules[f["id"]] = "list"
        elif f["id"] in static:
            rules[f["id"]] = "static"
        else:
            rules[f["id"]] = "none"
    info = {
        "median": median,
        "min_boxes": params.menu_min_boxes,
        "cover": params.menu_cover,
        "list": LIST_MENU_COVER if lists else float("inf"),
        "jump": float("inf") if lists else params.menu_jump * median,
        "static_s": STATIC_MIN_S if feed_times is not None else float("inf"),
        "static_bare_s": STATIC_BARE_MIN_S if feed_times is not None else float("inf"),
        "static_diff": STATIC_MAX_DIFF,
    }
    return rules, info


def _log_menu_frames(
    name: str,
    frames: list[dict[str, Any]],
    rules: dict[Any, str],
    info: dict[str, float],
    events: list[dict[str, Any]],
    feed_times: set[float] | None = None,
    text_spans: list[tuple[float, float]] | None = None,
) -> None:
    """One DEBUG line per frame that is a menu or has an OCR sighting: its numbers and
    the rule that fired, no OCR text (logs get shared). ``menu text`` tells whether
    the frame's still part shows new menu text (``-``: not in such a part)."""
    seen: dict[float, int] = {}
    for e in events:
        for t in e["_times"]:
            seen[t] = seen.get(t, 0) + 1
    still: dict[Any, float] = {}  # how long each frame's still stretch lasts
    for run in still_stretches(frames):
        for f in run:
            still[f["id"]] = run[-1]["t"] - run[0]["t"] if f.get("still") is not None else 0.0
    text: dict[Any, str] = {}
    if feed_times is not None:
        for part, shown in static_parts(frames, feed_times, text_spans or []):
            text.update((p["id"], "yes" if shown else "no") for p in part)

    def off(v: float, fmt: str) -> str:
        return "off" if v == float("inf") else f">= {v:{fmt}}"

    jump, lists = off(info["jump"], ".0f"), off(info["list"], ".2f")
    static = (
        "off"
        if info["static_s"] == float("inf")
        else f">= {info['static_s']:.1f}s with menu text, >= {info['static_bare_s']:.1f}s without"
    )
    for f in frames:
        rule, n = rules[f["id"]], seen.get(f["t"], 0)
        if rule == "none" and not n:
            continue
        logger.debug(
            "%s menu test %.1fs: %d boxes (min %d), centre cover %.2f (cover >= %.2f, list %s), "
            "median %.1f, jump %s, still %.1fs (static %s), menu text %s, %d OCR events -> %s",
            name,
            f["t"],
            len(f["boxes"]),
            info["min_boxes"],
            _centre_cover(f),
            info["cover"],
            lists,
            info["median"],
            jump,
            still[f["id"]],
            static,
            text.get(f["id"], "-"),
            n,
            rule,
        )


# --------------------------------------------------------------------------- reader


# Region kinds whose interpretation never looks at pixels: their reads keep no band.
TEXT_ONLY_KINDS = ("counter", "banner", "notice", "log", "bigtext")


@dataclass
class FrameRec:
    """One decoded frame as :class:`OcrReader` saw it: all :class:`OcrInterpretation`
    needs, and what the raw OCR cache keeps (:mod:`cutpoint.events.ocr_raw`).

    ``regions`` has a record per region the reader looked at: ``{"g": "det"}`` (a
    "det" gate call; the layout did not change and has no text rows), ``{"g":
    "keep"}`` (the layout did not change: the text is still on screen) or ``{"g":
    "read", "items": [[box, text, score], ...]}`` (an OCR call). Records of "det"
    gate regions other than feeds carry the gate's ``boxes`` (reference px ``[x0,
    y0, x1, y1]``, for the menu test); reads in regions whose kind looks at pixels
    (not :data:`TEXT_ONLY_KINDS`) carry ``rows``, the crop rows ``[a, b]`` their
    boxes cover, whose pixels are stacked in ``bands``. A region skipped by its
    ``every``, or that its "mask" gate did not let through, has no record.
    ``thumb`` is the :func:`_thumb_bgr` thumbnail (None without "det" gates).
    """

    n: int
    t: float
    thumb: np.ndarray | None = None
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    bands: dict[str, np.ndarray] = field(default_factory=dict)


def _plain(v: Any) -> Any:
    """A numpy number as a Python one (JSON; the same arithmetic live and replayed)."""
    return v.item() if hasattr(v, "item") else v


def _band_rows(items: list[list], h: int) -> list[list[int]]:
    """The rows of a crop of height ``h`` that the boxes of OCR ``items`` cover,
    merged: all the rows the colour rules read (:func:`_box_crop`, :func:`colour_runs`
    and :func:`_merge_adjacent`, whose boxes keep the rows of their parts)."""
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


def _union(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """The decoded rectangle: the bounding box of the region ``rects`` (even size)."""
    x0 = min(x for x, _, _, _ in rects)
    y0 = min(y for _, y, _, _ in rects)
    x1 = max(x + w for x, _, w, _ in rects)
    y1 = max(y + h for _, y, _, h in rects)
    return x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2


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
        self.sx, self.sy = 1920 / size[0], 1080 / size[1]
        self.thumbs = any(r.gate == "det" for r in regions)  # the static-screen test
        self.last_mask: dict[str, np.ndarray | None] = {r.name: None for r in regions}
        self.last_layout: dict[str, list[tuple[int, ...]] | None] = {r.name: None for r in regions}
        self.last_rows: dict[str, list[tuple[float, float, float, float, float]]] = {}

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
            sx, sy = self.sx, self.sy
            rec["boxes"] = [
                [
                    _plain((rx + min(p[0] for p in b)) * sx),
                    _plain((ry + min(p[1] for p in b)) * sy),
                    _plain((rx + max(p[0] for p in b)) * sx),
                    _plain((ry + max(p[1] for p in b)) * sy),
                ]
                for b in boxes
            ]
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

    def read(self, n: int, t: float, frame: np.ndarray) -> tuple[FrameRec, dict[str, np.ndarray]]:
        """The record of frame ``n`` (time ``t``) and the crops of the regions read."""
        rec = FrameRec(n, t, _thumb_bgr(frame) if self.thumbs else None)
        crops: dict[str, np.ndarray] = {}
        ux, uy, _, _ = self.union
        min_score = self.params.min_text_score
        for r in self.regions:
            if r.every > 1 and n % r.every:
                continue
            x, y, w, h = self.rects[r.name]
            crop = frame[y - uy : y - uy + h, x - ux : x - ux + w]
            if r.gate == "det":
                g: dict[str, Any] = {}
                rec.regions[r.name] = g
                if not self._det_changed(r, crop, g):
                    g["g"] = "keep" if self.last_layout[r.name] else "det"
                    continue
                rows = self.last_rows[r.name]
                if r.det_scale < 1.0:
                    # the cheap gate's boxes clip the ends of rows: re-detect at
                    # full resolution for exact row boxes (~35 ms)
                    rows = group_rows(
                        self.engine.detect(crop), r.min_text_h * self.scale, 10 * self.scale
                    )
                items = self.engine.read_rows(
                    crop,
                    [(y0, x0, y1, x1) for _, x0, x1, y0, y1 in rows],
                    min_score,
                    pad=6 * self.scale,
                )
            else:
                mask = _small(hud_mask(crop))
                if mask.mean() < r.min_fill:
                    self.last_mask[r.name] = mask
                    continue
                prev = self.last_mask[r.name]
                if prev is not None and np.mean(np.abs(mask - prev) > 0.25) < r.min_change:
                    continue
                self.last_mask[r.name] = mask
                items = self.engine(crop, min_score)
                g = rec.regions[r.name] = {}
            g["g"] = "read"
            g["items"] = [
                [[[_plain(v) for v in p] for p in box], str(text), float(score)]
                for box, text, score in items
            ]
            if r.kind not in TEXT_ONLY_KINDS and (rows := _band_rows(g["items"], crop.shape[0])):
                g["rows"] = rows
                rec.bands[r.name] = np.concatenate([crop[a:b] for a, b in rows])
            crops[r.name] = crop
        return rec, crops

    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """The reader's state: JSON (layouts, rows) and arrays (masks)."""
        masks = {
            f"mask{i}": m
            for i, r in enumerate(self.regions)
            if (m := self.last_mask[r.name]) is not None
        }
        return {"layout": self.last_layout, "rows": self.last_rows}, masks

    def restore(self, state: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        """Continue from a :meth:`state`."""
        layout = state["layout"]
        self.last_layout = {
            r.name: None if layout.get(r.name) is None else [tuple(v) for v in layout[r.name]]
            for r in self.regions
        }
        self.last_rows = {k: [tuple(v) for v in rows] for k, rows in state["rows"].items()}
        self.last_mask = {r.name: arrays.get(f"mask{i}") for i, r in enumerate(self.regions)}


def _chunk(
    recs: list[FrameRec],
    regions: tuple[Region, ...],
    state: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> ocr_raw.Chunk:
    """Frame records (and the reader's state after them) as a raw OCR cache chunk."""
    index = {r.name: i for i, r in enumerate(regions)}
    images = {f"{rec.n}-{index[name]}": band for rec in recs for name, band in rec.bands.items()}
    if recs and recs[0].thumb is not None:
        arrays = arrays | {"thumbs": np.stack([rec.thumb for rec in recs])}
    frames = [{"n": rec.n, "t": rec.t, "r": rec.regions} for rec in recs]
    return ocr_raw.Chunk(frames, state, images, arrays)


def _records(chunk: ocr_raw.Chunk, regions: tuple[Region, ...]) -> list[FrameRec]:
    """The frame records of a raw OCR cache chunk."""
    index = {r.name: i for i, r in enumerate(regions)}
    thumbs = chunk.arrays.get("thumbs")
    return [
        FrameRec(
            f["n"],
            f["t"],
            None if thumbs is None else thumbs[i],
            f["r"],
            {
                name: chunk.images[f"{f['n']}-{index[name]}"]
                for name, g in f["r"].items()
                if "rows" in g
            },
        )
        for i, f in enumerate(chunk.frames)
    ]


# --------------------------------------------------------------------------- interpretation


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
        self.interp = Interpreter(params, height / 1080, preset)
        rects = {r.name: scaled(r, width, height) for r in regions}
        union = _union(list(rects.values()))
        self.shapes = {name: _crop_shape(rect, union) for name, rect in rects.items()}
        self.stats = stats
        stats |= {"frames": 0, "ocr_calls": {r.name: 0 for r in regions}}
        det = any(r.gate == "det" for r in regions)
        if det:
            stats["det_calls"] = {r.name: 0 for r in regions if r.gate == "det"}
        # menu test (JSON presets): the det boxes of every region in reference px, per frame
        self.menus = self.interp.generic and det
        self.det_boxes: dict[str, list[list[float]]] = {}
        self.frame_boxes: list[dict[str, Any]] = []
        # static screens (see STATIC_THUMB): only when the decoded HUD area sees enough of the frame
        area = union[2] * union[3] / max(1, width * height)
        self.still_runs = StillRuns(params.fps) if self.menus and area >= STATIC_MIN_AREA else None
        if self.menus:
            stats["static_area"] = round(area, 2)
        self.region_events: dict[str, list[dict[str, Any]]] = {}

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
                stats["det_calls"][r.name] += 1
                if self.menus and r.kind != "feed":
                    self.det_boxes[r.name] = g.get("boxes", [])
                if g["g"] == "keep":  # same text still on screen
                    tracker.keep_alive(self.region_events.get(r.name, []), rec.t)
                if g["g"] != "read":
                    continue
            stats["ocr_calls"][r.name] += 1
            img = _band_image(self.shapes[r.name], g.get("rows", []), rec.bands.get(r.name))
            items = [(box, text, score) for box, text, score in g["items"]]
            n_before = len(tracker.events)
            if r.gate == "det":
                tracker.reset_touched()
                interp.handle(rec.t, r, img, items)
                self.region_events[r.name] = list(tracker.touched)
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

    def finish(self) -> list[dict[str, Any]]:
        """The events, after the menu frames' sightings are dropped (JSON presets)."""
        tracker, stats = self.interp.tracker, self.stats
        if self.menus:
            feeds = {r.name for r in self.regions if r.kind == "feed"}

            def feed_event(e: dict[str, Any]) -> bool:
                """Feed sightings and kills (kills come from the feed region): never dropped.
                This assumes menus do not cover the feed region; a scoreboard drawn over it
                is kept."""
                return e.get("region") in feeds or e.get("type") == "kill"

            tracked = tracker.events
            feed_times = {t for e in tracked if feed_event(e) for t in e["_times"]}
            # new menu text: when texts that are no preset phrase were first read, until when
            # shown; ``end`` moves only on the frames a region is read, so it is padded by one
            # read interval
            step = {r.name: r.every / self.interp.params.fps for r in self.regions}
            text_spans = [
                (e["_times"][0], e["end"] + step.get(e.get("region"), 0.0))
                for e in tracked
                if e.get("unknown")
            ]
            if self.still_runs is None:
                feed_times, text_spans = None, None
            frames = self.frame_boxes
            rules, menu_info = menu_rules(frames, feed_times=feed_times, text_spans=text_spans)
            if logger.isEnabledFor(logging.DEBUG):
                _log_menu_frames(
                    self.src_name, frames, rules, menu_info, tracked, feed_times, text_spans
                )
            menu = {f["t"] for f in frames if rules[f["id"]] != "none"}
            tracker.drop_sightings(menu, keep=feed_event)
            stats["menu_frames"] = len(menu)
            stats["static_frames"] = sum(r == "static" for r in rules.values())
        events = tracker.finish()
        for e in events:
            e["kind"] = "ocr"
        return events


def detect_ocr_events(
    src: Source,
    params: OcrParams,
    engine: OcrEngine | None,
    force: bool = False,
    debug_dir: Path | None = None,
    avoid: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    make_engine: Callable[[], OcrEngine] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OCR HUD events for one source (cached per source + parameters).

    JSON presets (not ``"builtin"``) drop what menu and scoreboard frames show
    (:func:`menu_times`): every frame records the text boxes the "det" gates
    last detected in its regions other than feeds (a region keeps its boxes on
    the frames it is skipped) and the still run of the decoded HUD area
    (:func:`static_frames`; when that area is at least :data:`STATIC_MIN_AREA` of
    the frame), and after the pass the sightings in menu frames
    are forgotten.
    No OCR runs for this; ``stats["menu_frames"]`` counts them, of which
    ``stats["static_frames"]`` held still.

    The read (:class:`OcrReader`) and its interpretation (:class:`OcrInterpretation`)
    are apart: the reader's records go into a raw OCR cache
    (:mod:`cutpoint.events.ocr_raw`), so when only the interpretation changed (a
    :data:`GENERIC_VERSION` bump) the result is replayed from it in seconds, with
    no OCR call and no ``engine`` (``stats["replayed"]`` is then True; ``ocr_calls``
    are those of the cached read), and an interrupted read resumes after its last
    finished chunk.

    Args:
        src: Source video (read once, sequentially).
        params: OCR parameters.
        engine: Loaded OCR engine, or None to only read the caches.
        force: Ignore the cache and drop the raw OCR cache (read the footage again).
        debug_dir: If set, save the region crop of every new event as a PNG (the
            footage is always read then).
        avoid: Time ranges never decoded (bad sectors); part of the cache key.
        on_progress: Called with the share (0-1) of the source decoded so far,
            at most once per percent.
        make_engine: Instead of ``engine``: makes the engine only when the footage is
            read (a replay or a cache hit loads no OCR model). Its engine must use
            :data:`OCR_DET_SIDE`, which the raw cache key assumes.

    Returns:
        (events, stats).
    """
    game = game_key_for_path(str(src.path), params.game, params.project)
    preset = load_preset(game, params.project)
    regions = params.regions or (preset.all_regions() if preset else ())
    if not regions:
        logger.info(
            "%s: no HUD preset for %r; skipping OCR (run with --discover-regions)", src.name, game
        )
        return [], {"skipped": "no preset", "game": game}
    avoid = sorted(avoid or [])
    newest = str(preset.feed.get("newest", "bottom")) if preset is not None else "bottom"
    raw = ocr_raw.RawStore(
        src.cdir,
        ocr_raw.raw_key(
            params.fps,
            [r.key() for r in regions],
            newest,
            params.min_text_score,
            avoid,
            getattr(engine, "det_side", OCR_DET_SIDE),
        ),
    )
    key = params.key(game, preset) | ({"avoid": avoid} if avoid else {})
    key["raw"] = cache.params_hash(raw.key)
    cpath = layer_cache_path(src.cdir, "ocr", key)
    if not force and (hit := cache.load_json(cpath)) is not None:
        return hit["events"], hit.get("stats", {})
    can_read = engine is not None or make_engine is not None
    if force and can_read:  # without an engine nothing would read it again
        raw.drop()
    meta = None if debug_dir else raw.meta()
    if meta is not None and meta.get("failed"):
        meta = None  # a read with a part that gave no frames: read it all again
    if not can_read and not (meta and meta.get("complete")):
        return [], {"skipped": "not cached"}

    if meta is not None:
        width, height, duration = (meta["info"][k] for k in ("width", "height", "duration"))
    else:
        info = probe(src.io_path)
        width, height, duration = info.width, info.height, info.duration
    rects = {r.name: scaled(r, width, height) for r in regions}
    union = _union(list(rects.values()))
    stats: dict[str, Any] = {"game": game, "hwaccel": False}
    if preset is not None and preset.path is not None:
        stats["preset"] = str(preset.path)
    interp = OcrInterpretation(src.name, params, preset, regions, (width, height), stats)
    t_start = time.perf_counter()
    segments = keep_segments(duration, avoid)
    if avoid:
        stats["skipped_ranges"] = [[round(a, 1), round(b, 1)] for a, b in avoid]
    shown = [-1]  # the last percent reported to on_progress

    def report(t: float) -> None:
        pct = int(100 * min(1.0, t / duration)) if duration > 0 else 0
        if on_progress is not None and pct > shown[0]:
            shown[0] = pct
            on_progress(pct / 100)

    # replay the finished raw chunks: all of a complete read, or those an interrupted one left
    done, last = 0, None
    if meta is not None:
        if meta.get("complete"):
            raw.touch()  # in use: another process's cleanup keeps it
        for chunk in raw.chunks(meta):
            for rec in _records(chunk, regions):
                interp.add(rec)
                report(rec.t)
            done, last = done + 1, chunk
    if meta is not None and meta.get("complete") and done == meta["chunks"]:
        raw.touch()
        stats |= meta.get("stats", {})  # hwaccel and read errors of the read
        stats["replayed"] = True
    elif not can_read:  # a damaged raw read
        return [], {"skipped": "not cached"}
    else:
        if engine is None:
            engine = make_engine()
        reader = OcrReader(
            regions, rects, union, engine, params, height / 1080, newest, (width, height)
        )
        # where the read goes on: segment, frame in it, frame number, decoder, read errors
        pos: dict[str, Any] = {"seg": 0, "idx": 0, "frame_no": 0, "hw": params.hwaccel}
        pos |= {"hwaccel": False, "read_errors": 0}
        if last is not None:
            reader.restore(last.state["reader"], last.arrays)
            pos = last.state["pos"]
            stats["hwaccel"] = pos["hwaccel"]
            logger.info(
                "%s: resuming the OCR read after %d frames (raw OCR cache)",
                src.name,
                stats["frames"],
            )
        writer = raw if raw.writable() else None
        pending: list[FrameRec] = []

        def write(step: Callable[..., None], *args: Any) -> None:
            """A raw cache write; on a disk error the read goes on without the raw cache."""
            nonlocal writer
            try:
                step(*args)
            except OSError as err:
                logger.warning("%s: raw OCR cache not written: %s", src.name, err)
                writer = None

        frame_no, hw = pos["frame_no"], pos["hw"]
        failed = bool(pos.get("failed"))  # a segment gave no frames: the read stays incomplete
        errors: list[str] = []
        all_errors: list[str] = []

        def flush(seg: int, idx: int) -> None:
            """Write the pending records as a chunk; the read resumes at frame ``idx`` of
            segment ``seg``. The first flush starts the raw read (an interrupt before it
            leaves no folder)."""
            if writer is not None and not writer.started:
                write(writer.start, done, {"width": width, "height": height, "duration": duration})
            if writer is None or not pending:
                return
            state, arrays = reader.state()
            n_errors = sum(bool(READ_ERROR_RE.search(e)) for e in all_errors + errors)
            at = {"seg": seg, "idx": idx, "frame_no": frame_no, "hw": hw}
            at |= {"hwaccel": stats["hwaccel"], "read_errors": pos["read_errors"] + n_errors}
            at["failed"] = failed
            write(writer.add, _chunk(pending, regions, {"reader": state, "pos": at}, arrays))
            pending.clear()

        def save_crop(out: Path, t: float, r: Region, crop: np.ndarray) -> None:
            """Save the region crop of a new event as a PNG in ``out``."""
            import cv2  # noqa: PLC0415

            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{src.path.stem[-24:]}_{t:07.1f}_{r.name}.png"), crop)

        by_name = {r.name: r for r in regions}
        for si, (seg_a, seg_b) in enumerate(segments):
            if si < pos["seg"]:
                continue
            k0 = pos["idx"] if si == pos["seg"] else 0  # frames of the segment already read
            start = seg_a + k0 / params.fps
            if start >= seg_b:
                continue
            while True:
                errors.clear()
                k = k0
                try:
                    for t, frame in iter_frames(
                        src.io_path,
                        params.fps,
                        union,
                        hw,
                        errors,
                        start,
                        None if seg_b >= duration else seg_b - start,
                    ):
                        if k0:  # resumed: the time stamps of an uninterrupted read
                            t = seg_a + k / params.fps
                        k += 1
                        rec, crops = reader.read(frame_no, t, frame)
                        frame_no += 1
                        for name in interp.add(rec):
                            if debug_dir:
                                save_crop(debug_dir, t, by_name[name], crops[name])
                        if writer is not None:
                            pending.append(rec)
                            if len(pending) >= ocr_raw.CHUNK_FRAMES:
                                flush(si, k)
                        report(t)
                    stats["hwaccel"] = hw
                    break
                except RuntimeError as err:
                    if k0 and k == k0 and start > seg_b - 1.0:
                        # resumed at the very end of the segment: nothing was left
                        logger.debug("%s: nothing left after %.1fs: %s", src.name, start, err)
                        break
                    if hw:
                        logger.warning(
                            "%s: hwaccel decode failed (%s); retrying on CPU", src.name, err
                        )
                        hw = False
                        continue
                    logger.warning("%s: no frames in %.0f-%.0fs: %s", src.name, seg_a, seg_b, err)
                    failed = True
                    break
            all_errors += errors
        errors.clear()
        read_errors = [e for e in all_errors if READ_ERROR_RE.search(e)]
        if read_errors:
            logger.warning(
                "%s: %d read errors during OCR decode (e.g. %s)",
                src.name,
                len(read_errors),
                read_errors[0],
            )
        stats["read_errors"] = pos["read_errors"] + len(read_errors)
        flush(len(segments), 0)
        if writer is not None and failed:
            logger.info("%s: raw OCR cache left incomplete (a part gave no frames)", src.name)
            write(writer.fail)
        elif writer is not None:
            write(writer.finish, {k: stats[k] for k in ("hwaccel", "read_errors")})
    stats["decoded_s"] = round(stats["frames"] / params.fps, 1)
    stats["elapsed_s"] = round(time.perf_counter() - t_start, 2)
    events = interp.finish()
    if stats.get("replayed"):
        logger.info(
            "%s: %d OCR events, %d frames, replayed from the raw OCR cache: 0 live OCR calls "
            "(the cached read made %s), %.1fs",
            src.name,
            len(events),
            stats["frames"],
            stats["ocr_calls"],
            stats["elapsed_s"],
        )
    else:
        logger.info(
            "%s: %d OCR events, %d frames, OCR calls %s, %.1fs",
            src.name,
            len(events),
            stats["frames"],
            stats["ocr_calls"],
            stats["elapsed_s"],
        )
    cache.save_json(
        cpath, {"source": str(src.path), "params": key, "stats": stats, "events": events}
    )
    return events, stats
