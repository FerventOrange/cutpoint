"""Layer 1b: on-screen events (kill feed, round banners, notices) via OCR.

One sequential ffmpeg pass per source decodes only the bounding box of the
game's HUD regions at a low frame rate (the probed GPU decoder when it works) and
pipes raw BGR frames to numpy. Each region is OCR'd with RapidOCR (CPU) only
when its gate lets it through: the text layout RapidOCR detects changed ("det")
or its "HUD pixel" mask (white text, blue/orange boxes) changed ("mask"), which
skips the vast majority of frames.

Events: ``{t, end, type, text, confidence, region, ...}`` with type one of
``kill`` (and the other feed verbs), ``banner``, ``notice``, ``bigtext``, ``multikill`` and
``score`` (a "score" region's number stepped by one, :class:`ScoreTracker`).
"""

from __future__ import annotations

import bisect
import contextlib
import difflib
import functools
import logging
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, memguard, state
from cutpoint.events import ocr_raw
from cutpoint.events.common import Source, layer_cache_path
from cutpoint.events.discover import (
    SCORE_SLOT_PX,
    DiscoverParams,
    _centre_cover,
    gameplay_frames,
    score_value,
)
from cutpoint.events.menu_spans import merge_menu_frames
from cutpoint.events.regions import HudPreset, Region, game_key_for_path, load_preset, scaled
from cutpoint.ffmpeg_utils import READ_ERROR_RE, decode_raw, decoder_for, probe
from cutpoint.imageio import write_png

logger = logging.getLogger(__name__)

MIN_TEXT_SCORE = 0.6  # recognised text below this score is dropped
FUZZY = 0.8  # SequenceMatcher ratio for phrase / dedupe matches
PERSIST_S = 8.0  # an entry seen again within this is the same event


@dataclass
class OcrParams:
    """Tunable parameters for :func:`detect_ocr_events` (all go into the cache key)."""

    fps: float = 3.0
    game: str = "auto"
    player_tags: tuple[str, ...] = ()  # config.json ``player_tags`` (or ``--player``)
    hwaccel: bool = True
    project: Path | None = None  # for project-local presets (not part of the key)
    # skip the det gate of a region that has not changed (DET_SKIP_*); config.json
    # ``ocr_det_skip`` or ``--ocr-det-skip``. In the raw read's key, so in the key through it
    det_skip: bool = False

    def key(self, game: str | None, preset: HudPreset | None = None) -> dict[str, Any]:
        """Cache key (hwaccel does not change results, so it is left out).

        :func:`detect_ocr_events` adds the hash of the raw read's key (``"raw"``,
        which includes the RapidOCR version), so results are read again after a
        RapidOCR upgrade.
        """
        key = {
            "fps": self.fps,
            "game": game,
            "tags": list(self.player_tags),
            "min_text_score": MIN_TEXT_SCORE,
            "fuzzy": FUZZY,
            "persist_s": PERSIST_S,
            "regions": [r.key() for r in (preset.all_regions() if preset else ())],
            "v": 4,  # 4: no built-in presets, no ``extra``
        }
        if preset is not None:
            key["preset"] = preset.key() | {"interp": GENERIC_VERSION}
        return key


# bump when the generic (JSON preset) interpretation changes; 7: menu frames give no
# events (:func:`menu_times`), multi-word phrases and feed kill words match whole words only;
# 8: static screens are menus too (:func:`static_frames`); 9: menu frames keep feed sightings
# and kills, a static screen without new menu text must hold still STATIC_BARE_MIN_S;
# 10: new menu text is STATIC_TEXT_LINES texts that stay for STATIC_TEXT_SHARE of the part;
# 11: screens with TEXT_LINES lines of new text that stay, TEXT_TOGETHER first read together,
# are menus too (:func:`text_frames`)
# 12: "row_notices" feeds without team colours read icon kill lines ("name [icon] name",
# FeedParser._icon_line) and other rows as notices; icon kill events keep their most complete
# names and are dropped on menu frames
# 13: menu frames are recorded as stats["menu_spans"] (#440)
# 14: gaps up to MENU_FILL_S between box- or text-rule menu frames are menus too (#409)
# 15: a static screen without new menu text needs menu frames within STATIC_ANCHOR_S (#554)
# 16: feeds learned row forms from the footage, a clock-stamped log's wrapped rows are joined,
# the default kill words are generic only and player tags tolerate OCR variants
# (:func:`is_player`)
# 17: feeds learn nothing from the footage, so no kills learned at 16 are reused
# 18: a discovered feed region that reads like stream chat (:func:`chat_label`) gives no events
# 19: a chat row (:func:`_is_chat`) gives no kill, assist or crash on any feed, not only in a log
# 20: reads of one kill row are one kill (:func:`merge_kill_reads`), a banner text that stays on a
# screen is one banner (:func:`collapse_banner_bursts`), a recurring system line is no kill
# (:meth:`Interpreter.drop_notice_kills`) and tags match tighter and in glued names
# (:func:`player_side`)
# 21: banner bursts remove a phrase OCR reads the same wrong way at its best fuzzy
# match (:func:`collapse_banner_bursts`)
# 22: big-text lines that come back as short pop-ups all through a source are flagged
# ``announce`` (:func:`learn_announcements`), short lines with digits included
# 23: text-rule menu frames are further than STATIC_FEED_GUARD_S from feed activity (#858)
# 24: the rows a feed reads as notices are one line of new menu text at most, however many
# are on screen (:func:`feed_stacks`, #891)
# 25: ... but the rows a feed reads first in one frame are lines of their own (a screen's text in
# the feed's area, #891)
GENERIC_VERSION = 25

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


# Letters OCR reads for one another, folded to one form when names are compared
_LOOKALIKES = str.maketrans("0158L", "OISBI")
TAG_MIN_FUZZY = 5  # shorter player tags must match a whole word of the name
# A tag read with OCR errors (:func:`_find_tag`): the edits (a dropped, added or, in a long tag,
# another letter) allowed by the tag's length (letters and digits), so a name that is another
# player's does not pass for the tag. Measured on gameplay footage (a 13-letter tag, misread in 11
# of 42 kill rows): one letter read as another (r as n), one or two dropped, never more than two.
TAG_EDIT_STEPS = (7, 12, 16)  # a tag this long or longer allows one, two, three edits more
TAG_SUBSTITUTE_MIN = 12  # a shorter tag allows no letter read as another ("Moonrock", "Moonreck")


def _fold(text: str) -> str:
    """Upper-case letters and digits, no spaces, look-alikes folded ("O0" and "OO" are one
    form), for comparing two reads of a name."""
    return norm(text).replace(" ", "").translate(_LOOKALIKES)


def _tag_edits(n_tag: int) -> int:
    """The edits a tag of ``n_tag`` letters and digits may be read with (:data:`TAG_EDIT_STEPS`)."""
    return sum(n_tag >= size for size in TAG_EDIT_STEPS)


@state.register_cache
@functools.lru_cache(maxsize=8192)
def _find_tag(text: str, tag: str) -> tuple[int, int] | None:
    """Where the folded ``tag`` is in the folded ``text``, as (start, end), allowing the OCR
    errors of :func:`_tag_edits`; None if nowhere. The closest stretch wins (the earlier on a
    tie)."""
    at = text.find(tag)
    if at >= 0:
        return at, at + len(tag)
    allowed = _tag_edits(len(tag))
    if not allowed:
        return None
    sub = 1 if len(tag) >= TAG_SUBSTITUTE_MIN else 2  # else a letter read as another costs two
    best: tuple[int, int, int] | None = None
    for size in range(len(tag) - allowed, len(tag) + allowed + 1):
        for start in range(len(text) - size + 1):
            row = list(range(len(tag) + 1))
            for i in range(1, size + 1):
                prev, row = row, [i]
                for j, c in enumerate(tag, 1):
                    same = text[start + i - 1] == c
                    row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (0 if same else sub)))
            if row[-1] <= allowed and (best is None or row[-1] < best[0]):
                best = (row[-1], start, start + size)
    return None if best is None else (best[1], best[2])


def _fuzzy_player(n: str, words: set[str], t: str) -> bool:
    """Whether the folded name ``n`` holds the folded tag ``t`` read with OCR errors. A read
    that needs an edit counts only when it is the whole name or a whole word of it, or the tag
    is at least :data:`TAG_SUBSTITUTE_MIN` letters long: "Dunmorrow" is no read of "Dunmore" and
    "Phoenician" none of "Phoenix". A missed credit is better than a false one."""
    found = _find_tag(n, t)
    if not found:
        return False
    if n.find(t) >= 0 or len(t) >= TAG_SUBSTITUTE_MIN:
        return True
    return any(_find_tag(w, t) == (0, len(w)) for w in (n, *words))


def is_player(name: str, tags: tuple[str, ...]) -> bool:
    """Whether an OCR'd name is one of the user's tags, allowing OCR variants: spaces and
    punctuation are ignored ("Mari _gold"), look-alike letters are one (:func:`_fold`:
    "M00nrock"), a clan tag or a weapon may be read with the name, and a long enough tag may
    be read with a letter dropped or added (:func:`_find_tag`: "Marigld" for "Marigold"; a
    long tag also with a letter read as another). A tag of fewer than :data:`TAG_MIN_FUZZY`
    letters and digits must be a whole word: "Kip" is no part of "Skipper"."""
    n = _fold(name)
    words = {_fold(w) for w in re.findall(r"[^\W_]+", _CAMEL_RE.sub(" ", name))}
    for tag in tags:
        t = _fold(tag)
        if not (n and t):
            continue
        if t == n or t in words or t == _fold(split_name(name)[0]):
            return True
        if len(t) >= TAG_MIN_FUZZY and _fuzzy_player(n, words, t):
            return True
    return False


def player_side(actor: str, target: str, tags: tuple[str, ...]) -> tuple[bool, bool]:
    """Whether the actor and the target of a feed row are one of the user's tags
    (:func:`is_player` per side; both are when a player kills themself). When neither is, but
    the two names joined hold a tag of 7 letters or more (OCR cut the row between the names at
    the wrong place: "Stonhar,QQ" and "uillmere77" for "Stonhar,Q" and "Quillmere77"), the side
    that holds most of the tag is; neither is when the sides hold equal parts. A shorter tag is
    not looked for in two joined names: "Mike" and "Vince" are no "Kevin"."""
    p_actor = bool(actor.strip()) and is_player(actor, tags)
    p_target = is_player(target, tags)
    if p_actor or p_target or not actor.strip():
        return p_actor, p_target
    na, nt = _fold(actor), _fold(target)
    for tag in tags:
        t = _fold(tag)
        found = _find_tag(na + nt, t) if _tag_edits(len(t)) >= 1 else None
        if found and found[0] < len(na) < found[1]:  # a match inside one name is no tag
            on_actor = max(0, min(found[1], len(na)) - found[0])
            on_target = max(0, found[1] - max(found[0], len(na)))
            if on_actor != on_target:
                return on_actor > on_target, on_target > on_actor
    return False, False


# --------------------------------------------------------------------------- pixels


def hud_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of likely HUD pixels: white text, blue and orange boxes."""
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    mn = np.minimum(np.minimum(b, g), r)
    mx = np.maximum(np.maximum(b, g), r)
    white = (mn > 200) & (mx - mn < 40)
    blue = (b > 150) & (r < 120) & (b - r > 80)
    orange = (r > 200) & (g > 90) & (g < 190) & (b < 100)
    return white | blue | orange


_HUES = (  # upper hue limits (degrees) of the colour names
    (15, "red"),
    (38, "orange"),
    (68, "yellow"),
    (160, "green"),
    (200, "cyan"),
    (250, "blue"),
    (320, "purple"),
    (361, "red"),
)


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
    return next((name for limit, name in _HUES if h < limit), None)


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


# Icon kill lines without a kill word ("name [weapon icon] name"): the two names are split
# by a stretch of ICON_GAP text heights or more. OCR boxes split there, or one box spans
# the whole line; then the names sit on coloured columns (their box or their text: at least
# ICON_FILL of a column's pixels have saturation >= ICON_SAT and value >= ICON_VAL, in 0-255;
# white text and dark icon boxes have neither) and the icon gap is the widest run of other
# columns, with a text height of coloured columns on both sides. The characters are shared
# out by the widths on both sides and the cut moved to the nearest word boundary within
# ICON_SNAP characters (a space wins a tie). Measured on one VOD (solid and see-through
# name boxes): most kill-line reads split with both names right, and of the other rows only a
# menu tab row, whose right part _icon_name rejects.
ICON_GAP = 1.0
ICON_SAT = 60
ICON_VAL = 90
ICON_FILL = 0.3
ICON_SNAP = 2.5
_ICON_NAME_BAD = re.compile(r"[:%/]")
# Feed rows read as notices (``row_notices``) that stay longer than this are no feed lines but
# text that stays in the feed's area (a lobby line: "We're estimating your chances to win..."
# for 49 s in one VOD); feed lines scroll out within seconds (kill lines and notices
# there: up to 10 s).
ROW_NOTICE_MAX_S = 20.0
# Kill lines tracked by their two names: icon lines and the kills of a clock-stamped log
LINE_ENTRIES = ("icon", "text")
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
# Two reads of an icon kill line's name are one name when the shorter shares a run of
# ICON_PART of its characters with the longer (a cut read with noise at the cut)
ICON_PART = 0.8
# ... or when the shorter is within one edit per ICON_EDIT_CHARS of its characters (at least
# one) of a start of the longer (misread letters: "NAVA" for "NOVA", "DRARNATI" of
# "DRAMATICRAIL39")
ICON_EDIT_CHARS = 4
# Reads of one kill row that were not matched as it was tracked (a name split at the wrong
# place, a different cut of it) are one kill when the later read starts within KILL_ROW_S of the
# earlier one's start (a feed row lasts about 5 s: the median kill event of the measured footage
# is 4.7 s) and no later than KILL_REREAD_GAP_S after its last sighting (a row is missed for a
# few frames while it slides in or fades; at 3 frames a second that is a second or two).
KILL_ROW_S = 6.0
KILL_REREAD_GAP_S = 2.0
SAME_NAME = 0.9  # a side this similar is the same name, so a mismatch on the other side is a kill
# A banner phrase that comes back BANNER_BURST_MIN times running, each time within
# BANNER_BURST_GAP_S of the last, on the same longer line of text, is text that stays on a
# screen (a scoreboard with player names, a tournament table), not announcements. Such banners
# on the measured gameplay were at least 45 s apart and never three in a row within a minute.
BANNER_BURST_GAP_S = 60.0
BANNER_BURST_MIN = 3
# Learned announcements (:func:`learn_announcements`): a word of a big-text line no phrase
# matched, or a short line with digits and a word read as its shape ("4 VS 5" is "# VS #"; a
# bare number such as "+100" or "12:34" is never learned), that comes back as a short pop-up
# all through a source is an announcement. Notice-sized lines never are: item and action
# prompts recur there. Appearances: reads of
# the key that start within ANNOUNCE_BRIEF_S of the previous one's end are one (one pop-up read
# in two regions); one longer than ANNOUNCE_LONG_S is a label or a place name, not a pop-up. A
# key is learned with at least ANNOUNCE_MIN_N short appearances, at most ANNOUNCE_LONG_SHARE of
# all its appearances long, a median gap of at least ANNOUNCE_MIN_GAP_S between the short ones,
# and short ones in at least ANNOUNCE_MIN_PARTS different ANNOUNCE_PART_S stretches (so at
# least 30 minutes of one source).
ANNOUNCE_MIN_N = 10
ANNOUNCE_BRIEF_S = 8.0
ANNOUNCE_LONG_S = 30.0
ANNOUNCE_LONG_SHARE = 0.2
ANNOUNCE_MIN_GAP_S = 30.0
ANNOUNCE_PART_S = 600.0
ANNOUNCE_MIN_PARTS = 4
_ANNOUNCE_WORD_RE = re.compile(r"[^\W\d_]{4,}")  # a word of 4 letters or more, any script
# Score readout (:class:`ScoreTracker`): the number in a "score" region (a round score, a
# counter; :func:`~cutpoint.events.discover.score_value`) is read per slot (boxes whose centres
# lie within half of SCORE_SLOT_PX across). A value counts once read SCORE_CONFIRM times in a
# row at SCORE_MIN_CONF or more. A change is an event when the new value is one more or one
# less than the old, the old one held SCORE_HOLD_S, and the HUD showed no number for at most
# SCORE_MAX_GAP_S between them; each slot keeps the direction of its first change (a counter
# that counts down is fine, a step back afterwards is no event). Anything else is taken as the
# new value silently (a reset between matches). A value that comes back within SCORE_HOLD_S is
# a misread. A slot whose value changes more often than every SCORE_HOLD_S (median) or that
# shows more than SCORE_MAX_VALUES values is no score (a clock, ammo): no events from it.
# Nor is a slot that once moved by two or more in its direction to a value that held
# SCORE_HOLD_S, with at most SCORE_MAX_GAP_S between them (:func:`_score_jumps`): a
# score goes up or down one point at a time, a count of items (gadgets placed, charges used)
# drops by several at once; on a 60 min recording a gadget counter beside the team badge
# was taken for a round score that way (#872). A jump the other way is a reset and stays fine.
SCORE_CONFIRM = 2
SCORE_MIN_CONF = 0.7
SCORE_HOLD_S = 15.0
SCORE_MAX_GAP_S = 60.0
SCORE_MAX_VALUES = 12


def column_colours(bgr: np.ndarray) -> list[str | None]:
    """The colour name of each column of a box crop (:data:`_HUES`, the median hue of its
    coloured pixels), or None when fewer than :data:`ICON_FILL` of its pixels are coloured
    (saturation >= :data:`ICON_SAT`, value >= :data:`ICON_VAL`)."""
    import cv2  # noqa: PLC0415

    if bgr.size == 0:
        return []
    hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
    coloured = (hsv[..., 1] >= ICON_SAT) & (hsv[..., 2] >= ICON_VAL)
    out: list[str | None] = []
    for x in range(hsv.shape[1]):
        m = coloured[:, x]
        if m.sum() < ICON_FILL * len(m):
            out.append(None)
            continue
        h = float(np.median(hsv[m, x, 0])) * 2  # OpenCV hue is 0-180
        out.append(next((n for lim, n in _HUES if h < lim), None))
    return out


def _icon_cut(text: str, k: float) -> int:
    """Where to cut ``text`` into two names near character ``k``: the nearest space or word
    boundary (:func:`_is_bound`) within :data:`ICON_SNAP` characters where the right name does
    not start with punctuation (a space wins a tie), else ``k`` rounded."""
    near = [i for i in range(1, len(text)) if abs(i - k) <= ICON_SNAP]

    def space(i: int) -> bool:
        return text[i].isspace() or text[i - 1].isspace()

    bounds = [
        i
        for i in near
        if space(i) or (_is_bound(text, i) and (text[i].isalnum() or text[i].isspace()))
    ]
    return min(bounds, key=lambda i: (abs(i - k), not space(i))) if bounds else round(k)


def icon_split(bgr: np.ndarray, text: str) -> tuple[str, str, str | None, str | None] | None:
    """``(left name, right name, left colour, right colour)`` of one OCR box ``text`` (crop
    ``bgr``) that spans an icon kill line, or None.

    The icon gap is the widest run of uncoloured columns (:func:`column_colours`) at least
    :data:`ICON_GAP` crop heights wide with a crop height of coloured columns on both
    sides; each name's colour is the most common one on its side. The text is cut at the
    share of the width left of the gap (:func:`_icon_cut`)."""
    h, w = bgr.shape[:2]
    cols = column_colours(bgr)
    on = np.array([c is not None for c in cols])
    best: tuple[int, int] | None = None
    x = 0
    while x < w:
        end = x
        while end < w and on[end] == on[x]:
            end += 1
        if (
            not on[x]
            and end - x >= ICON_GAP * h
            and on[:x].sum() >= h
            and on[end:].sum() >= h
            and (best is None or end - x > best[1] - best[0])
        ):
            best = (x, end)
        x = end
    if best is None:
        return None
    a, b = best
    t = text.strip()
    cut = _icon_cut(t, len(t) * a / max(1, a + w - b))
    left, right = t[:cut].strip(), t[cut:].strip()
    if not left or not right:
        return None

    def common(side: list[str | None]) -> str | None:
        named = [c for c in side if c]
        return max(set(named), key=named.count) if named else None

    return left, right, common(cols[:a]), common(cols[b:])


def _icon_name(text: str, min_len: int) -> bool:
    """Whether ``text`` reads as one name of an icon kill line: one or two words (OCR
    splits names), ``min_len`` letters and digits or more, half of them letters, no
    ":" "%" "/" (readouts such as "FPS:106" or "3/4") and no bracket without its other half
    (an item cut off in a chat line; a clan tag has both)."""
    n = _alnum_len(text)
    return (
        n >= min_len
        and len(text.split()) <= 2
        and sum(ch.isalpha() for ch in text) >= 0.5 * n
        and not _ICON_NAME_BAD.search(text)
        and text.count("[") == text.count("]")
    )


def _small(mask: np.ndarray, k: int = 4) -> np.ndarray:
    h, w = mask.shape[0] // k * k, mask.shape[1] // k * k
    return mask[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


# --------------------------------------------------------------------------- decoding


class OcrCancelled(BaseException):
    """A read with a ``cancel`` event was stopped: nothing after the last finished raw
    chunk is kept. A BaseException, so the read's ``except RuntimeError`` (a failed
    decode pass) never takes it for a decode error."""


def ocr_vf(fps: float, rect: tuple[int, int, int, int]) -> str:
    """The filter chain of the OCR decode: ``fps`` frames a second of ``rect`` (x, y, w, h)."""
    x, y, w, h = rect
    return f"fps={fps},crop={w}:{h}:{x}:{y}"


# The decode hook of detect_ocr_events (a shared decode, #375): (rect, chain, stderr lines,
# exit codes) -> the frames as iter_frames yields them, or None to decode as usual.
DecodeHook = Callable[
    [tuple[int, int, int, int], str, list[str], list[int]],
    Iterator[tuple[float, np.ndarray]] | None,
]


def iter_frames(
    path: Path,
    fps: float,
    rect: tuple[int, int, int, int],
    hw: bool,
    errors: list[str],
    start: float = 0.0,
    dur: float | None = None,
    cancel: threading.Event | None = None,
    exit_codes: list[int] | None = None,
):
    """Yield ``(t, frame)`` for the cropped rectangle; stderr lines go to ``errors``.

    Only ``[start, start + dur]`` is decoded when given; ``t`` is source time. With
    ``cancel`` (a background read), ffmpeg runs out of the console's reach of Ctrl+C and is
    killed once ``cancel`` is set (``decode_raw(isolate=True)``); ``exit_codes`` gets a
    non-zero exit of ffmpeg that the decode did not cause.

    Raises:
        RuntimeError: If ffmpeg produced no frames at all (an
            :class:`~cutpoint.errors.FFmpegError`; the caller may retry
            without hwaccel).
    """
    _x, _y, w, h = rect
    yield from decode_raw(
        path,
        size=(w, h),
        vf=ocr_vf(fps, rect),
        pix_fmt="bgr24",
        start=start if start > 0 else None,
        duration=dur,
        stamp_fps=fps,
        hwaccel="auto" if hw else None,
        fallback_cpu=False,  # the caller switches to the CPU for the remaining segments
        stderr_lines=errors,
        isolate=cancel is not None,
        cancel=cancel,
        exit_codes=exit_codes,
    )


# --------------------------------------------------------------------------- tracking


def _two_hues(e: dict[str, Any]) -> bool:
    """Whether a kill line's names have two team colours, which labels and menu rows do not."""
    hues = {e.get("killer_hue"), e.get("victim_hue")}
    return len(hues) == 2 and not hues & {None, "white"}


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

    @staticmethod
    def _nested(a: str | None, b: str | None) -> bool:
        """Whether one name (normalized, spaces ignored) is part of the other: a read cut
        while the line slides in or fades ("BOXE" of "BOXER7"), allowing OCR noise at the cut
        (at least :data:`ICON_PART` of the shorter name in one run: "PDRAMATIC" of
        "DRAMATICRAIL39")."""
        na, nb = norm(a or "").replace(" ", ""), norm(b or "").replace(" ", "")
        if not (na and nb):
            return False
        m = difflib.SequenceMatcher(None, na, nb, autojunk=False)
        run = m.find_longest_match(0, len(na), 0, len(nb)).size
        return run >= ICON_PART * min(len(na), len(nb))

    @staticmethod
    def _misread(a: str | None, b: str | None) -> bool:
        """Whether one name (normalized, spaces ignored) is the other, or its start (a cut
        read), but for misread letters: the shorter one is within one edit per
        :data:`ICON_EDIT_CHARS` characters (at least one) of a start of the longer. A short
        name misread by one letter ("NAVA" for "NOVA", "KLP" for "KIP") or a cut read with
        "rn" for "m" ("DRARNATI" of "DRAMATICRAIL39") is no other name."""
        na, nb = norm(a or "").replace(" ", ""), norm(b or "").replace(" ", "")
        if not (na and nb):
            return False
        short, long = sorted((na, nb), key=len)
        row = list(range(len(long) + 1))  # edits from "" to each start of ``long``
        for i, ch in enumerate(short, 1):
            prev, row = row, [i]
            for j, c in enumerate(long, 1):
                row.append(min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ch != c)))
        return min(row) <= max(1, len(short) // ICON_EDIT_CHARS)

    def _same_line(self, e: dict[str, Any], fields: dict[str, Any]) -> bool:
        """Whether an icon kill event and a sighting are one line: per side the names are
        nested (:meth:`_nested`), misread (:meth:`_misread`) or similar. Two lines of one
        killer ("Alder > Kip", "Alder > Boxer7") are two kills even when a long killer name
        makes the keys similar."""
        return all(
            self._nested(e.get(s), fields.get(s))
            or self._misread(e.get(s), fields.get(s))
            or similar(e.get(s) or "", fields.get(s) or "") >= self.fuzzy
            for s in ("killer", "victim")
        )

    @staticmethod
    def _most_complete(e: dict[str, Any], fields: dict[str, Any]) -> None:
        """An icon kill line's names and colours from its most complete read, whatever its
        confidence: each read (killer, victim) of the line is counted in ``_reads``, and the
        names are those of the read with the most letters and digits over both names among
        the reads seen at least twice, or among all when none repeats (a read with junk
        glued on, seen once, does not win; both names from one read, so two bad splits are
        not combined; a tie keeps the earlier read). Only reads of one line
        (:meth:`_same_line`) get here. Measured on one VOD: a cut read ("dramati", 0.95)
        often has a higher confidence than the whole line ("dramatic_rail39", 0.91)."""
        read = (fields.get("killer"), fields.get("victim"))
        hues = {k: fields.get(k) for k in ("killer_hue", "victim_hue")}
        # a kill first read with no entry (a log row before its clock showed) has no reads yet
        e.setdefault("_reads", {}).setdefault(read, {"n": 0} | hues)["n"] += 1
        reads = list(e["_reads"].items())
        pool = [r for r in reads if r[1]["n"] >= 2] or reads
        (killer, victim), best = max(
            pool, key=lambda r: _alnum_len(r[0][0] or "") + _alnum_len(r[0][1] or "")
        )
        e |= {"killer": killer, "victim": victim, "text": f"{killer} > {victim}"}
        e |= {k: best[k] for k in ("killer_hue", "victim_hue")}

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
            icon = e.get("entry") in LINE_ENTRIES and fields.get("entry") in LINE_ENTRIES
            if icon:  # one line by its names (:meth:`_same_line`): the key is the killer's
                match = (
                    e["_key"] == key
                    if dup
                    else self._same_line(e, fields) or (exclusive and self._same_victim(e, fields))
                )
            else:
                match = (
                    e["_key"] == key
                    or (not dup and similar(e["_key"], key) >= self.fuzzy)
                    or (exclusive and self._same_victim(e, fields))
                )
            if e["type"] == etype and match:
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
                if icon:
                    self._most_complete(e, fields)
                    # the names may come from a read of lower confidence: keep a player
                    # flag found in any read
                    for k in ("player", "player_kill", "player_death"):
                        if k in fields:
                            e[k] = bool(e.get(k)) or bool(fields[k])
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
        if fields.get("entry") in LINE_ENTRIES:  # the reads of a kill line (_most_complete)
            read = (fields.get("killer"), fields.get("victim"))
            hues = {k: fields.get(k) for k in ("killer_hue", "victim_hue")}
            ev["_reads"] = {read: {"n": 1} | hues}
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
            if e["sightings"] >= min_sightings or e["type"] == "banner":
                out.append({k: v for k, v in e.items() if not k.startswith("_")})
        return out


def _same_kill_read(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two kill events are reads of one feed row: per side the names are nested,
    misread or similar (:meth:`EventTracker._same_line`'s rules), or the two names joined
    are similar (:data:`FUZZY`; OCR can cut the row between the names at another place, so
    neither side matches). Never when the two were seen in one frame (``_times``, which the
    tracker keeps until it finishes): the tracker keeps two rows on screen at once apart on
    purpose. Never when one side is the same player and the other clearly another (a killer
    with two victims, a victim of two killers): that is two kills. A wrong cut changes both
    sides, so a side that is nearly the same name (:data:`SAME_NAME`) while the other does not
    match is two kills too ("Alderwick > Boxer77" and "Alderwick > Boxer12")."""
    if set(a.get("_times", ())) & set(b.get("_times", ())):
        return False
    sides = [(_fold(a.get(s) or ""), _fold(b.get(s) or "")) for s in ("killer", "victim")]
    if not all(x and y for x, y in sides):
        return False
    ratios = [difflib.SequenceMatcher(None, x, y).ratio() for x, y in sides]
    if max(ratios) >= 0.9 and min(ratios) < 0.5:
        return False
    if all(
        EventTracker._nested(x, y) or EventTracker._misread(x, y) or r >= FUZZY
        for (x, y), r in zip(sides, ratios, strict=True)
    ):
        return True
    if max(ratios) >= SAME_NAME:
        return False
    joined = difflib.SequenceMatcher(None, sides[0][0] + sides[1][0], sides[0][1] + sides[1][1])
    return joined.ratio() >= FUZZY


def merge_kill_reads(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the kill events that are reads of one feed row into one (the events not merged
    away are returned in their order; the merged one is updated in place). Run on the
    tracker's events, which still have their ``_times`` (:meth:`EventTracker.finish` removes
    them).

    The tracker joins sightings of a row when their names match side by side, which fails for
    OCR variants and for a row split at another place ("Mari > goldKip" and "Marigold > Kip"),
    so one row can leave two kills, which would count twice. Reads of one row are
    close in time (:data:`KILL_ROW_S`, :data:`KILL_REREAD_GAP_S`) and have the same names
    (:func:`_same_kill_read`). The read kept is that of the most letters and digits over both
    names among those seen at least twice, or among all when none was (a cut read often has a
    higher confidence than the whole row, as in :meth:`EventTracker._most_complete`; the more
    confident one on a tie). It starts at the first read's start and ends at the last read's end,
    counts all the sightings and keeps any player flag of the reads. Two kills of one killer, or
    of one victim, with another name on the other side stay two."""
    kills = sorted((e for e in events if e.get("type") == "kill"), key=lambda e: e["t"])
    groups: list[list[dict[str, Any]]] = []
    live: list[list[dict[str, Any]]] = []  # the groups that can still take a read
    for e in kills:
        live = [
            g
            for g in live
            if e["t"] - g[0]["t"] <= KILL_ROW_S
            and e["t"] - max(x["end"] for x in g) <= KILL_REREAD_GAP_S
        ]
        for g in live:
            if any(_same_kill_read(x, e) for x in g) and not any(
                set(x.get("_times", ())) & set(e.get("_times", ())) for x in g
            ):
                g.append(e)
                break
        else:
            groups.append([e])
            live.append(groups[-1])
    dropped: set[int] = set()
    for g in groups:
        if len(g) == 1:
            continue
        seen_twice = [x for x in g if x.get("sightings", 1) >= 2] or g
        best = max(
            seen_twice,
            key=lambda x: (
                _alnum_len(x.get("killer") or "") + _alnum_len(x.get("victim") or ""),
                x.get("confidence") or 0,
            ),
        )
        best["t"] = min(x["t"] for x in g)
        best["end"] = max(x["end"] for x in g)
        best["sightings"] = sum(x.get("sightings", 1) for x in g)
        if "_times" in best:
            best["_times"] = sorted({t for x in g for t in x.get("_times", ())})
        for k in ("player", "player_kill", "player_death"):
            if k in best:
                best[k] = any(bool(x.get(k)) for x in g)
        dropped |= {id(x) for x in g if x is not best}
    return [e for e in events if id(e) not in dropped]


def collapse_banner_bursts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first banner of each burst of text that stays on a screen (:data:`BANNER_BURST_MIN`
    or more in a row, each within :data:`BANNER_BURST_GAP_S` of the one before). Such text is a
    scoreboard or table that holds a phrase in a player name: its banners have a ``raw`` line
    longer than the phrase and the same line every time (:func:`similar`). Banners that
    announce something come in the same words each time but with nothing else on the line or
    with another ``detail`` each time ("ELIMINATED" and a name), so they are never collapsed,
    however often they come. The ones kept have ``recurrences``, the number of banners dropped
    after them. Other event types and the banners not in such a burst are returned as they
    were. Measured on a held-out scoreboard whose player names include "ACE": 14 banners within
    6 minutes, 10-50 s apart, all read from one screen."""
    texts: dict[int, str] = {}

    def screen(e: dict[str, Any]) -> str:
        """The text on the banner's line besides the phrase when it holds a name, else ""."""
        if id(e) not in texts:
            texts[id(e)] = _line_rest(e.get("raw") or "", e.get("text") or "")
        return texts[id(e)]

    banners = [e for e in events if e.get("type") == "banner"]
    dropped: set[int] = set()
    for run in _held_runs(banners, lambda e: e.get("text") or "", screen):
        run[0]["recurrences"] = len(run) - 1
        dropped |= {id(e) for e in run[1:]}
    return [e for e in events if id(e) not in dropped]


def _line_rest(raw: str, phrase: str) -> str:
    """The text of the line ``raw`` besides ``phrase`` when it holds a name (three letters in a
    row), else "" (:func:`collapse_banner_bursts`)."""
    line = norm(raw).replace(" ", "")
    phrase = norm(phrase).replace(" ", "")
    # a score ("+100") is no screen, a name is; a phrase OCR reads the same wrong way every
    # time is removed at its best fuzzy match, or the leftover would pass as a name
    rest = line.replace(phrase, "", 1)
    if phrase and phrase not in line:
        best, at = 0.0, 0  # strictly better only, so a tie keeps the leftmost window
        for i in range(len(line) - len(phrase) + 1):
            ratio = difflib.SequenceMatcher(None, line[i : i + len(phrase)], phrase).ratio()
            if ratio > best:
                best, at = ratio, i
        if best >= FUZZY:
            rest = line[:at] + line[at + len(phrase) :]
    return rest if re.search(r"[A-Za-z]{3}", rest) else ""


def _held_runs(
    events: list[dict[str, Any]],
    phrase: Callable[[dict[str, Any]], str],
    screen: Callable[[dict[str, Any]], str],
) -> list[list[dict[str, Any]]]:
    """The runs of ``events`` that are text staying on a screen: :data:`BANNER_BURST_MIN` or
    more in a row, each within :data:`BANNER_BURST_GAP_S` of the one before, with a similar
    ``phrase`` and the same ``screen`` text besides it (never an empty one)."""
    runs: list[list[dict[str, Any]]] = []
    for e in sorted(events, key=lambda e: e["t"]):
        for run in runs:
            if (
                e["t"] - run[-1]["t"] <= BANNER_BURST_GAP_S
                and screen(e)
                and similar(phrase(run[0]), phrase(e)) >= FUZZY
                and similar(screen(run[0]), screen(e)) >= FUZZY
            ):
                run.append(e)
                break
        else:
            runs.append([e])
    return [run for run in runs if len(run) >= BANNER_BURST_MIN and all(screen(e) for e in run)]


def _announce_keys(text: str) -> list[str]:
    """The keys a line can be learned by (:func:`learn_announcements`): its words of 4 letters
    or more, upper-case, any script; for a line with none, its shape: its upper-case words with
    each run of digits as "#" ("4 VS 5" is "# VS #"), when it has a word of 2 letters or more
    (none for a bare number: "+100", "12:34", "2 - 3"; none for one letter beside digits, a
    misread counter such as "x10")."""
    words = [w.upper() for w in _ANNOUNCE_WORD_RE.findall(text)]
    if words:
        return list(dict.fromkeys(words))
    shape = " ".join(re.sub(r"\d+", "#", w) for w in re.findall(r"[^\W_]+", text.upper()))
    return [shape] if re.search(r"[^\W\d_]{2,}", shape) else []


def _announce_screen(e: dict[str, Any], key: str, learned: set[str], fixed: set[str]) -> str:
    """The text on an event's line besides the announcement, for the held-screen test: for a
    word ``key``, the rest of the line without the ``fixed`` words of the announcement (tokens of
    :func:`norm`) and every ``learned`` word in it when that holds a name (three letters in a
    row, as :func:`_line_rest`), else ""; for a shape, its digits."""
    text = e.get("text") or ""
    if not _ANNOUNCE_WORD_RE.fullmatch(key):
        return " ".join(re.findall(r"\d+", text))
    line = "".join(w for w in norm(text).split() if w not in fixed)
    for word in sorted(learned.intersection(_announce_keys(text)), key=len, reverse=True):
        line = line.replace(norm(word).replace(" ", ""), "", 1) if norm(word) else line
    return line if re.search(r"[A-Za-z]{3}", line) else ""


def learn_announcements(
    events: list[dict[str, Any]], extra: Sequence[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Flag the big-text lines no phrase matched that come back as short pop-ups all through
    the source: ``announce`` True and ``announce_words`` (the learned keys in the line,
    :func:`_announce_keys`). Any such line counts, a one-word line too, but not notices (an item
    or action prompt recurs there as often) and not banners. ``extra``: the short big-text lines
    with digits that the interpretation keeps out of its events (:meth:`Interpreter.handle`);
    the flagged ones are added to the events returned, the others dropped.

    Per key, reads that start within :data:`ANNOUNCE_BRIEF_S` of the last one's end are one
    appearance, and a key is learned under the gate of :data:`ANNOUNCE_MIN_N`. The first event
    of each short appearance is flagged, but text that stays on a screen
    (:func:`_held_runs` with the key as the phrase and :func:`_announce_screen`: the line without
    every learned key and the words in at least half of the key's lines, so another word of the
    announcement is no name) keeps only its first. The event type stays as it was."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for e in (*events, *extra):
        if e.get("type") == "bigtext" and e.get("unknown"):
            for key in _announce_keys(e.get("text") or ""):
                by_key.setdefault(key, []).append(e)
    shown: dict[str, list[dict[str, Any]]] = {}  # learned key -> first events of its appearances
    for key, found in by_key.items():
        apps: list[list[dict[str, Any]]] = []
        end = 0.0
        for e in sorted(found, key=lambda e: e["t"]):
            if apps and e["t"] - end <= ANNOUNCE_BRIEF_S:
                apps[-1].append(e)
                end = max(end, e["end"])
            else:
                apps.append([e])
                end = e["end"]
        brief = [a for a in apps if max(x["end"] for x in a) - a[0]["t"] <= ANNOUNCE_LONG_S]
        starts = [a[0]["t"] for a in brief]
        if (
            len(brief) < ANNOUNCE_MIN_N
            or len(apps) - len(brief) > ANNOUNCE_LONG_SHARE * len(apps)
            or float(np.median([b - a for a, b in pairwise(starts)])) < ANNOUNCE_MIN_GAP_S
            or len({int(t // ANNOUNCE_PART_S) for t in starts}) < ANNOUNCE_MIN_PARTS
        ):
            continue
        shown[key] = [a[0] for a in brief]
    learned = set(shown)
    for key, firsts in shown.items():
        # the announcement's own words, short ones too ("YOU", "RED"): tokens in at least half
        # of the key's lines; a name beside it repeats only within a burst
        counts: dict[str, int] = {}
        for e in by_key[key]:
            for w in set(norm(e.get("text") or "").split()):
                counts[w] = counts.get(w, 0) + 1
        fixed = {w for w, n in counts.items() if 2 * n >= len(by_key[key])}
        runs = _held_runs(
            firsts,
            lambda e, k=key: k,
            lambda e, k=key, f=fixed: _announce_screen(e, k, learned, f),
        )
        held = {id(e) for run in runs for e in run[1:]}
        for e in firsts:
            if id(e) not in held:
                e["announce"] = True
                e.setdefault("announce_words", []).append(key)
    return [*events, *(e for e in extra if e.get("announce"))]


@dataclass
class _ScoreRun:
    """Reads of one value in a row (:class:`ScoreTracker`)."""

    value: int
    first: float
    last: float
    n: int
    conf: float  # the sum of the reads' confidences

    def held(self) -> float:
        return self.last - self.first


def _score_runs(reads: list[tuple[float, int, float]]) -> list[_ScoreRun]:
    """The confirmed values of one slot in time order: runs of :data:`SCORE_CONFIRM` reads or
    more of one value (each within :data:`SCORE_MAX_GAP_S` of the last), runs of one value
    joined across the misreads between them, and a value that comes back within
    :data:`SCORE_HOLD_S` joined across the other one (a misread too)."""
    runs: list[_ScoreRun] = []
    for t, v, c in reads:
        last = runs[-1] if runs else None
        if last is not None and last.value == v and t - last.last <= SCORE_MAX_GAP_S:
            last.last, last.n, last.conf = t, last.n + 1, last.conf + c
        else:
            runs.append(_ScoreRun(v, t, t, 1, c))
    out: list[_ScoreRun] = []
    for r in (r for r in runs if r.n >= SCORE_CONFIRM):
        if out and out[-1].value == r.value and r.first - out[-1].last <= SCORE_MAX_GAP_S:
            out[-1] = _join(out[-1], r)
        else:
            out.append(r)
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1 : i + 2]
        if a.value == c.value and b.held() < SCORE_HOLD_S and c.first - a.last <= SCORE_MAX_GAP_S:
            out[i - 1 : i + 2] = [_join(a, c)]
            i = max(1, i - 1)
        else:
            i += 1
    return out


def _join(a: _ScoreRun, b: _ScoreRun) -> _ScoreRun:
    return _ScoreRun(a.value, a.first, b.last, a.n + b.n, a.conf + b.conf)


class ScoreTracker:
    """The score readout of "score" regions: their reads per slot during the pass
    (:meth:`see`), the changes of the value as ``score`` events at its end (:meth:`finish`,
    rules at :data:`SCORE_SLOT_PX`)."""

    def __init__(self) -> None:
        self.reads: dict[tuple[str, int], list[tuple[float, int, float]]] = {}
        self._slots: dict[str, list[float]] = {}  # per region the slots' centre x (reference px)

    def see(
        self,
        t: float,
        region: str,
        items: list[tuple[list, str, float]],
        ref: tuple[float, float, float, float],
    ) -> None:
        """Record the numbers of one read of ``region`` at ``t`` (``ref``: the crop's left and
        top in frame px and the frame-to-reference factors): per slot the surest one."""
        rx, _, sx, _ = ref
        best: dict[int, tuple[int, float]] = {}
        for box, text, score in items:
            v = score_value(text)
            if v is None or score < SCORE_MIN_CONF:
                continue
            cx = (rx + (min(p[0] for p in box) + max(p[0] for p in box)) / 2) * sx
            slot = self._slot(region, cx)
            if slot not in best or score > best[slot][1]:
                best[slot] = (v, float(score))
        for slot, (v, c) in best.items():
            self.reads.setdefault((region, slot), []).append((t, v, c))

    def _slot(self, region: str, cx: float) -> int:
        """The slot of a number centred at ``cx`` (reference px): the first one within half
        of :data:`SCORE_SLOT_PX`, or a new one there; named by its centre x."""
        xs = self._slots.setdefault(region, [])
        for x in xs:
            if abs(cx - x) <= SCORE_SLOT_PX / 2:
                return round(x)
        xs.append(cx)
        return round(cx)

    def finish(self, menu: set[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The ``score`` events of the pass, reads at the ``menu`` frame times left out,
        and per slot (``<region>@<x>``) its ``reads``, ``values``, ``changes`` and whether it
        was ``disabled`` (no score: it changes too often, shows too many values or jumped in
        its direction, :func:`_score_jumps`)."""
        events: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        for (region, slot), reads in self.reads.items():
            runs = _score_runs([r for r in reads if r[0] not in menu])
            spans = [b.first - a.first for a, b in pairwise(runs)]
            disabled = (
                len({r.value for r in runs}) > SCORE_MAX_VALUES
                or (len(spans) >= 2 and float(np.median(spans)) < SCORE_HOLD_S)
                or _score_jumps(runs) > 0
            )
            found = [] if disabled else _score_changes(runs, region, slot)
            events += found
            stats[f"{region}@{slot}"] = {
                "reads": len(reads),
                "values": len({r.value for r in runs}),
                "changes": len(found),
                "disabled": disabled,
            }
        return events, stats


def _in_play(a: _ScoreRun, b: _ScoreRun) -> bool:
    """Whether the change from ``a`` to ``b`` happened in play: ``a`` held
    :data:`SCORE_HOLD_S` and at most :data:`SCORE_MAX_GAP_S` passed until ``b``."""
    return a.held() >= SCORE_HOLD_S and b.first - a.last <= SCORE_MAX_GAP_S


def _score_jumps(runs: list[_ScoreRun]) -> int:
    """How often one slot's value moved by two or more in its direction (the sign of its
    first in-play change by one, :func:`_in_play`) to a value that then held
    :data:`SCORE_HOLD_S`, at most :data:`SCORE_MAX_GAP_S` after the old one: a count of
    items, no score (:data:`SCORE_MAX_VALUES`). The old value need not have held: menus
    and the HUD hidden between rounds leave it few reads. None without a direction."""
    pairs = list(pairwise(runs))
    step = next(
        (b.value - a.value for a, b in pairs if abs(b.value - a.value) == 1 and _in_play(a, b)), 0
    )
    return sum(
        (b.value - a.value) * step >= 2
        and b.held() >= SCORE_HOLD_S
        and b.first - a.last <= SCORE_MAX_GAP_S
        for a, b in pairs
    )


def _score_changes(runs: list[_ScoreRun], region: str, slot: int) -> list[dict[str, Any]]:
    """The ``score`` events of one slot's confirmed values (:data:`SCORE_SLOT_PX`): at the last
    read of the old value, ending at the first read of the new one."""
    out = []
    step = 0  # the slot's direction, once set
    for a, b in pairwise(runs):
        d = b.value - a.value
        gap = b.first - a.last
        if abs(d) != 1 or not _in_play(a, b):
            continue  # a reset or a misread: the new value counts from here
        if step and d != step:
            continue  # a step back
        step = d
        out.append(
            {
                "t": round(a.last, 2),
                "end": round(b.first, 2),
                "type": "score",
                "text": f"{a.value} > {b.value}",
                "confidence": round(min(a.conf / a.n, b.conf / b.n), 3),
                "region": region,
                "before": a.value,
                "after": b.value,
                "seen_at": round(b.first, 2),
                "gap_s": round(gap, 2),
                "slot": slot,
            }
        )
    return out


# --------------------------------------------------------------------------- OCR logic


# onnxruntime intra-op threads for the OCR models (det, cls, rec). CPU OCR stops scaling at
# about 4 threads: 8 / 4 / 2 / 1 threads took 4.0 / 3.3 / 6.9 / 8.7 s per 1280 px frame on an
# 8-core CPU, so 4 is ~18% faster than the default (one per physical core). Capped at the CPU
# count so small machines are not oversubscribed. The output does not depend on it (#374).
OCR_THREADS = min(4, os.cpu_count() or 1)
# OCR lanes (#403): the thread budget split over engines that run at once. Crops and row strips
# are small and use intra-op threads poorly: on 4 vCPUs, 2 engines x 2 threads read a 300 s
# clip's HUD in 286 s instead of 345 s with 1 x 4 (4 x 1 was slower). The output does not
# depend on it.
OCR_LANES = 2 if OCR_THREADS >= 4 else 1


class OcrEngine:
    """Thin wrapper around RapidOCR (onnxruntime, CPU).

    With ``lanes`` > 1 it holds that many RapidOCR engines, which share the
    :data:`OCR_THREADS` budget, and :meth:`map` and :meth:`read_rows` run OCR calls on
    them at once. Each call runs on one engine, which gives the same result on any of
    them, so the output does not depend on the lanes.

    ``rec_batch`` sets how many text boxes a full call (:meth:`__call__`) recognises in one
    batch; None keeps RapidOCR's default (6). A batch pads every box to the widest one, so
    the text of a box can depend on the others: 1 recognises each box alone.
    """

    def __init__(self, lanes: int | None = None, rec_batch: int | None = None) -> None:
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

        memguard.check("ocr")
        self.lanes = OCR_LANES if lanes is None else lanes
        self.rec_batch = rec_batch
        extra = {} if rec_batch is None else {"rec_batch_num": rec_batch}

        # Default det config upscales the short side to 736 px ("limit_type: min"),
        # which makes wide HUD strips ~10x slower; cap the long side instead. RapidOCR
        # 1.4.4 then picks the limit from the image's long side (960 below 960 px, 1500
        # below 1500 px, else 2000), so crops up to 2000 px are never scaled and
        # det_limit_side_len would change nothing (#443).
        engines = [
            RapidOCR(
                det_limit_type="max",
                intra_op_num_threads=max(1, OCR_THREADS // self.lanes),
                **extra,
            )
            for _ in range(self.lanes)
        ]
        self._free: queue.SimpleQueue[Any] = queue.SimpleQueue()  # the engines not in use
        for e in engines:
            self._free.put(e)
        self._pool = ThreadPoolExecutor(self.lanes, "ocr-lane") if self.lanes > 1 else None

    @contextlib.contextmanager
    def _lane(self) -> Iterator[Any]:
        """An engine no other thread uses meanwhile."""
        e = self._free.get()
        try:
            yield e
        finally:
            self._free.put(e)

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        """``fn`` of every item, in order; on the lanes at once when there are several.
        ``fn`` may make OCR calls of this engine but not call :meth:`map` again."""
        if self._pool is None or len(items) < 2:
            return [fn(x) for x in items]
        return list(self._pool.map(fn, items))

    def detect(self, bgr: np.ndarray, scale: float = 1.0) -> list[list]:
        """Text boxes only (no recognition), used by the "det" gate.

        ``scale`` < 1 detects on a downscaled copy (much cheaper); boxes are
        returned in the coordinates of ``bgr``.
        """
        img = bgr
        if scale < 1.0:
            import cv2  # noqa: PLC0415

            img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        with self._lane() as e:
            result, _ = e(np.ascontiguousarray(img), use_cls=False, use_rec=False)
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
        strips = []
        for y0, x0, y1, x1 in rows:
            a, b = int(max(0, y0 - pad / 2)), int(min(h, y1 + pad / 2))
            c, d = int(max(0, x0 - 2 * pad)), int(min(w, x1 + pad))
            if b - a >= 6 and d - c >= 6:
                strips.append((a, b, c, d))

        def recognise(strip: tuple[int, int, int, int]) -> Any:
            a, b, c, d = strip
            with self._lane() as e:
                res, _ = e(
                    np.ascontiguousarray(bgr[a:b, c:d]), use_det=False, use_cls=False, use_rec=True
                )
            return res

        out = []
        for (a, b, c, d), res in zip(strips, self.map(recognise, strips), strict=True):
            for item in res or []:
                text, score = str(item[0]), float(item[1])
                if score >= min_score and text.strip():
                    out.append(([[c, a], [d, a], [d, b], [c, b]], text, score))
        return out

    def __call__(self, bgr: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        with self._lane() as e:
            result, _ = e(np.ascontiguousarray(bgr), use_cls=False)
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

        def same(a: str | None, b: str | None) -> bool:
            return tracker._nested(a, b) or tracker._misread(a, b)

        def label(e: dict[str, Any]) -> bool:
            if _two_hues(e):
                return False
            runs = _runs(e["_times"], STATIC_FEED_GUARD_S)
            if max(run[-1] - run[0] for run in runs) > ROW_NOTICE_MAX_S:
                return True
            repeats = [o for o in lines if tracker._same_line(e, o)]
            ids = {id(o) for o in repeats}
            names = (e["killer"], e["victim"])
            return len(repeats) >= LABEL_REPEATS and not any(
                same(n, o[side])
                for o in lines
                if id(o) not in ids
                for n in names
                for side in ("killer", "victim")
            )

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
        phrase is kept as ``detail`` ("by -BRoFS- Sinistro9121 (Strv m/40L)")."""
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
            letters = sum(ch.isalpha() for ch in line)
            etype = "notice" if region.kind == "notice" else "bigtext"
            tracker = self.tracker
            if letters < 4 or letters < 0.6 * len(line.replace(" ", "")):
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


_CLAN_RE = re.compile(r"^\s*(\[[^\]]{1,8}\]|\^[^\^]{1,8}\^|=[^=]{1,8}=|-[^-]{1,8}-)\s*")
# Generic words that work from the first row on; no game's phrases belong here (a preset's
# ``kill_words`` can list them)
DEFAULT_KILL_WORDS = {
    "destroyed": "kill",
    "shot down": "kill",
    "killed": "kill",
    "eliminated": "kill",
    "assist": "assist",
    "has crashed": "crash",
    "crashed": "crash",
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
# A chat line's mark before a row's kill word: "Kip: he got killed", "Kip (mod): he got killed",
# "Kip [mod]: he got killed" (OCR also reads it full width)
_SENTENCE_RE = re.compile(r"[^\s:;\uff1a\uff1b]\s*[:;\uff1a\uff1b]")
_CLOCK_RE = re.compile(r"\s*\d{1,2}:\d{2}(?!\d)\s*")
# A clock that starts a row, as OCR may read it: "2:20", "[2:20]", "(2:20)", "#2:20", "2: 20", "2;20"
_LEAD_CLOCK_RE = re.compile(
    r"\s*[\[(\uff08#]?\s*\d{1,3}\s*[:;\uff1a\uff1b]\s*\d{2}(?!\d)\s*[\])\uff09]?\s*"
)


def _is_chat(actor: str) -> bool:
    """Whether the text before a kill word reads as a chat line (:data:`_SENTENCE_RE`). What
    a row starts with is no chat mark: the clock of a log row that stayed alone ("2:20 Kip
    destroyed Alder") or a clan tag's own colon ("[A:B] Kip destroyed Alder"). A colon after
    the tag is one ("[mod]: he got killed"), so the tag stays as one character."""
    clock = _LEAD_CLOCK_RE.match(actor)
    text = actor[clock.end() :] if clock else actor
    return _SENTENCE_RE.search(_CLAN_RE.sub("_", text)) is not None


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


def _join_log_rows(lines: list[tuple]) -> tuple[list[tuple], set[int]]:
    """The entries of a clock-stamped log ("2:20 Kip destroyed" over "Alder (Tank)"): with two
    or more rows that start with a clock, each such row and the rows up to the next one are
    one entry, without the clock. Returns the lines and the indices of the log's lines, also
    those above its first clock (an entry cut off by the scroll)."""
    starts = [k for k, line in enumerate(lines) if _CLOCK_RE.match(line[0])]
    if len(starts) < 2:
        return lines, set()
    out = lines[: starts[0]]
    log = set(range(starts[0]))
    for a, b in zip(starts, [*starts[1:], len(lines)], strict=True):
        part = lines[a:b]
        text = " ".join(line[0] for line in part)
        conf = min(line[1] for line in part)
        log.add(len(out))
        out.append((_CLOCK_RE.sub("", text, count=1), conf, part[0][2], part[0][3]))
    return out, log


def _runs(times: list[float], gap: float) -> list[list[float]]:
    """The sorted ``times`` in runs without a step longer than ``gap``."""
    runs: list[list[float]] = []
    for t in sorted(times):
        if runs and t - runs[-1][-1] <= gap:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


class FeedParser:
    """Game-agnostic kill-feed parsing for "feed" regions.

    Two forms are recognised (both from one region crop):

    * a row containing a kill word ("A destroyed B", "A eliminated B"): actor before the
      word, target after it;
    * icon feeds without words, where an entry is a coloured *actor* line and
      a differently coloured *target* line (``"entry": "two_line"``)
      or two coloured names on one row (``"entry": "one_line"``);
      without a colour map, in a discovered icon feed (``"row_notices": true``), a
      row of two names split by an icon gap is a kill, left name first
      (:meth:`_icon_line`).

    A clock-stamped log's wrapped rows are joined first; only the kill words read them, as a
    log tells much more than kills.

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
        self.row_notices = bool(cfg.get("row_notices", False))
        self.notice_boxes: list[list] = []  # the last parse's notice rows (crop px)
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
        p_actor, p_target = player_side(_before_group(actor), _before_group(target), self.tags)
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

    def _icon_line(
        self, img: np.ndarray, names: list[tuple], line: str, conf: float
    ) -> tuple[dict[str, Any] | None, bool]:
        """A kill from an icon kill line ("name [weapon icon] name", no kill word) when the
        preset maps no colours to teams and its feed has ``row_notices`` (a feed discovery
        found as an icon feed): exactly two name boxes ``names`` at least
        :data:`ICON_GAP` text heights apart, or one box split by :func:`icon_split`. The
        left name is the killer (the usual order); both must read as names
        (:func:`_icon_name`). Returns the kill (or None) and whether the row has an icon
        gap. The colours (:func:`text_hue` of a box, or the name's side of
        :func:`icon_split`) are kept as ``killer_hue`` and ``victim_hue``."""
        if not names:
            return None, False
        if len(names) > 2:  # menu rows ("PLAY | STORE | CHARACTERS"), no name pair
            return None, False
        if len(names) == 2:
            (ba, ta, _), (bb, tb, _) = names
            h = float(np.median([max(p[1] for p in b) - min(p[1] for p in b) for b in (ba, bb)]))
            if min(p[0] for p in bb) - max(p[0] for p in ba) < ICON_GAP * h:
                return None, False
            ha, hb = text_hue(_box_crop(img, ba)), text_hue(_box_crop(img, bb))
        else:
            sp = icon_split(_box_crop(img, names[0][0]), names[0][1])
            if sp is None:
                return None, False
            ta, tb, ha, hb = sp
        ta, tb = ta.strip(), tb.strip()
        if not (_icon_name(ta, self.min_name) and _icon_name(tb, self.min_name)):
            return None, True
        ev = self._event(ta, tb, "kill", None, None, conf, line)
        if ev:
            ev |= {"killer_hue": ha, "victim_hue": hb, "entry": "icon"}
        return ev, True

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
        lines, log = _join_log_rows(lines)
        out: list[dict[str, Any]] = []
        used: set[int] = set()
        gapped: set[int] = set()  # rows split by an icon gap that made no kill
        for k, (line, conf, team, _row) in enumerate(lines):
            sp = self._word_split(line)
            if sp is None:
                continue
            actor, target, verb = sp
            if verb in ("kill", "assist", "crash") and _is_chat(actor):
                used.add(k)  # no other form reads it either
                continue
            if verb in ("crash",):
                actor, target = "", actor
            ev = self._event(actor, target, verb, team, None, conf, line)
            if ev:
                if k in log and verb == "kill":  # two kills of one actor stay on screen long
                    ev["entry"] = "text"
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
                if not self.colors:
                    if self.row_notices:  # a discovered icon feed
                        ev, gap = self._icon_line(img, names, line, conf)
                        if ev:
                            out.append(ev)
                            used.add(k)
                        elif gap:
                            gapped.add(k)
                    continue
                if len(names) < 2:
                    continue
                ka = self._team(img, [names[0]])
                kb = self._team(img, [names[-1]])
                if ka and kb and ka != kb and {ka, kb} <= {"ally", "enemy", "squad", "self"}:
                    ev = self._event(names[0][1], names[-1][1], "kill", ka, kb, conf, line)
                    if ev:
                        out.append(ev)
                        used.add(k)
        if self.row_notices:
            out += self._row_notices(lines, used | gapped | log)
        return out

    def _row_notices(self, lines: list[tuple], skip: set[int]) -> list[dict[str, Any]]:
        """Notices from feed rows that made no event and have no icon gap (not in
        ``skip``): a name and text ("Kip has left the game"), at least two words of which
        4 or more letters, 60% of the characters letters (as notice regions read). Their
        boxes are kept in ``notice_boxes`` for the menu test (kill rows stay out of it: a
        full feed is no menu)."""
        out = []
        self.notice_boxes = []
        for k, (line, conf, _, row) in enumerate(lines):
            letters = sum(ch.isalpha() for ch in line)
            if (
                k in skip
                or len(line.split()) < 2
                or letters < 4
                or letters < 0.6 * len(line.replace(" ", ""))
            ):
                continue
            h = float(np.median([max(p[1] for p in i[0]) - min(p[1] for p in i[0]) for i in row]))
            self.notice_boxes += [i[0] for i in row]
            out.append(
                {
                    "type": "notice",
                    "key": norm(line),
                    "text": line[:120],
                    "confidence": round(conf, 3),
                    "text_h": round(h / self.scale),
                    "unknown": True,
                }
            )
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


def link_teamkills(events: list[dict[str, Any]], generic_teamkills: bool = False) -> None:
    """(Re)compute the ``teamkill`` flag of kill events in place.

    A kill is a teamkill only when ``generic_teamkills`` (the preset's
    ``feed.teamkills``: it trusts its colours) and both name colours read as ally.
    Applied to cached results too, so the rule can change without re-OCR.
    """
    for e in events:
        if e.get("type") == "kill":
            e["teamkill"] = bool(
                generic_teamkills
                and e.get("killer_team") == "ally"
                and e.get("victim_team") == "ally"
            )


def derive_streaks(events: list[dict[str, Any]], window_s: float = 20.0) -> None:
    """Add ``multikill`` events for 2+ player kills within ``window_s`` (in place).

    Only feed kills (they carry ``verb``) are considered; a self-kill
    (``player_death`` set too) is no kill here. Idempotent: old derived events are dropped first.
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

# Static screens (rule "static"): garage, crew and loadout menus show as little text as a busy
# HUD (6-17 boxes, centre cover 0.04-0.12), so the box rules miss them, but they hold still
# while gameplay moves. Every OCR frame takes a STATIC_THUMB (w, h) grayscale thumbnail of the
# HUD area it already decoded, "score" regions left out (:func:`_still_rect`, :func:`_thumb_bgr`,
# :func:`_grey`; no extra decode) and :class:`StillRuns` gives it
# the id of its still run (``"still"`` in the frame record; no thumbnail is kept). A run goes
# on while thumbnails stay within STATIC_MAX_DIFF of its first frame (mean absolute difference
# in 0-255 grey levels: still video and its compression noise stay below 1, a moving game view
# changes by tens of levels, a menu cursor or a small animation by a few) and no frame is
# missing (a gap over STATIC_MAX_GAP_S, e.g. a skipped bad range, ends it). The frames within
# STATIC_FEED_GUARD_S of feed activity (a feed sighting or a kill, but not an icon kill first
# read within STATIC_FEED_GUARD_S of a still run's start: menu rows open with the screen) are
# taken out, so kills in still gameplay (scoped, spectating) are never dropped; the rest of a run is a static screen
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
# Stillness alone is weak evidence: a bare still part (no new menu text) also fits still
# gameplay (a respawn or spectator wait, a preparation phase), which held still for 11-34 s on
# hand-labelled footage of two games and was flagged as a menu. A bare part is a static
# screen only when menu evidence lies within STATIC_ANCHOR_S of it: a frame of another menu
# rule (``cover``, ``jump``, ``list``, ``text``); a still part with menu text is no evidence
# (a preparation phase after a still text screen would follow it). Menus come in runs (lobby,
# garage, matchmaking), so a garage next to its menu screens stays, an isolated still stretch
# in gameplay does not. A bare part that holds still for STATIC_LONG_S needs none: the longest
# still gameplay measured was 37.7 s (a preparation phase), a garage menu held 55.7 s.
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
STATIC_LONG_S = 45.0

# Text screens (rule "text"): lobbies, character select, map screens and result screens rarely
# hold still (animated backgrounds, cursors, countdowns end their still runs within seconds) and
# show their words in HUD regions in numbers no box rule sees, but many of their lines stay on
# screen. A frame is a text screen when at least TEXT_LINES lines of new menu text (the
# ``text_spans`` of :func:`static_parts`, so no preset phrase, number or feed line), each on
# screen for TEXT_STAY_S to TEXT_STAY_MAX_S, are on screen at once, at least TEXT_TOGETHER of
# them first read within STATIC_FEED_GUARD_S of each other, in a stretch of at least
# STATIC_MIN_FRAMES such frames over at least TEXT_MIN_S (frames at most STATIC_MAX_GAP_S
# apart, or 2 frame intervals at low OCR frame rates, as :class:`StillRuns`). A span runs from
# a line's first read to its last, so it bridges gaps up to PERSIST_S (8 s): a notice that
# keeps coming back counts as one line that stays. Gameplay text flashes (notices, kill lines,
# score popups) or is a few lasting labels that arrive one by one (a location, an objective,
# an overlay, a returning notice); a menu opens with many lines at once. Measured on footage of
# four games, gameplay showed at most 5 staying lines at once, never for TEXT_MIN_S and never
# TEXT_TOGETHER of them first read together; menus show up to 11. A line that is part of the
# HUD is no screen that opens and closes: a line on screen for over TEXT_STAY_MAX_S (a player
# list that stays all match: 7 lines over all 90 s of a long clip), and a line (one text in
# one region, all its spans: a list hidden by a scoreboard comes back as a new span) first
# read within TEXT_STAY_S of the read's start, on screen until within TEXT_STAY_S of its end
# (a missed read, a region read every few frames or a clip that opens on a fade) and for at
# least TEXT_HUD_SHARE of it, so a source shorter than TEXT_STAY_MAX_S is covered too.
# Like the static rule it runs only with ``feed_times`` and ``text_spans`` (the decoded HUD
# area is at least STATIC_MIN_AREA of the frame), and like it leaves out the frames within
# STATIC_FEED_GUARD_S of feed activity (#858): a busy gameplay moment (name tags, location labels
# and prompts piling up over seconds, the notices of a death) can reach TEXT_LINES staying lines,
# TEXT_TOGETHER of them first read together, for TEXT_MIN_S, but the kill feed goes on beside it;
# a lobby or result screen has no feed sightings or kills. Measured on footage of four games, this
# cleared a 3.3 s and a 28 s stretch of gameplay and kept every text-rule menu (two).
# A feed's rows are one stack, not lines of a screen (#891): a discovered icon feed
# (``row_notices``) reads the rows that are no kill line it can parse as notices, and a busy feed
# shows five or more of them, arriving together and on screen for 10 s or more, with no fresh
# read (so no feed sighting) while the feed holds still. Alone they passed as a text screen. The
# rows of one feed region that stay as lines of their own (:func:`staying_spans`: no short row,
# no HUD row) and overlap count as one line (:func:`feed_stacks`), so a menu whose text reaches
# into the feed's area keeps that line, a full feed adds one at most, and short rows back to back
# add none: a moment never has more lines than when each row counted alone. But a feed adds its
# rows one at a time (two at most in one frame in the gameplay measured, a double kill), while
# a screen whose text lies in the feed's area (a lobby list, a waiting dialog) opens with all of
# it in one frame: of every group of TEXT_TOGETHER or more rows first read in the same frame,
# the rows count each.
# Measured on footage of two games (four clips): this cleared a 12 s and a 7 s stretch of
# gameplay beside a busy feed, kept a lobby whose five lines sit in the feed's area (merged into
# one, it lost 11 s of menu) and every other menu span, and lost no event.
TEXT_LINES = 5
TEXT_TOGETHER = 3
TEXT_HUD_SHARE = 0.5
TEXT_STAY_S = 6.0
TEXT_STAY_MAX_S = 60.0
TEXT_MIN_S = 3.0

# Gaps between menu screens (rule "fill", #409): loading, matchmaking and "processing" screens
# between a lobby and the next screen show no text the box or text rules see, and an animated
# garage menu holds no still run, so a lobby-heavy source kept them as gameplay. Frames that no
# rule flags between two frames of the box or text rules (``cover``, ``jump``, ``list``,
# ``text``) at most MENU_FILL_S apart, with no feed activity between them, are menus too. A
# static frame is no end of a gap: still gameplay can pass the static rule, and a gap filled
# from it would grow into the gameplay next to it. Measured on footage of two games
# (hand-labelled menu time), recall rose from 0.75 to 0.89 and 0.54 to 0.64 with no new
# gameplay frame flagged; 15 s spans their loading screens, and longer gaps added none. Like
# the static rule it runs only with ``feed_times``.
MENU_FILL_S = 15.0
FILL_ENDS = ("cover", "jump", "list", "text")
STATIC_ANCHOR_S = MENU_FILL_S  # menu evidence for a bare still screen (see STATIC_THUMB)


def _thumb_bgr(frame: np.ndarray) -> np.ndarray:
    """The :data:`STATIC_THUMB` BGR thumbnail (uint8) of a decoded BGR frame, as the raw
    OCR cache keeps it."""
    import cv2  # noqa: PLC0415

    return cv2.resize(frame, STATIC_THUMB, interpolation=cv2.INTER_AREA)


def _still_rect(
    regions: tuple[Region, ...], rects: dict[str, tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    """The HUD area of the static-screen test (:data:`STATIC_THUMB`, :data:`STATIC_MIN_AREA`)
    in frame px: the :func:`_union` of the regions but the "score" ones (all of them when
    there are only those). A score box at a screen edge grows the decoded area; the
    thumbnails of the larger area put still runs and so the menu frames elsewhere (#872),
    which moved the text events of the other regions."""
    text = [rects[r.name] for r in regions if r.kind != "score"]
    return _union(text or list(rects.values()))


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
    anchor_times: Iterable[float] | None = None,
) -> set[Any]:
    """Ids of the frames of a static screen: a part of :func:`static_parts` with at
    least :data:`STATIC_MIN_FRAMES` frames that holds still for at least
    :data:`STATIC_MIN_S` when it shows new menu text (``text_spans``), else
    :data:`STATIC_BARE_MIN_S` and a time of ``anchor_times`` (frames another menu rule
    flagged) within :data:`STATIC_ANCHOR_S`. Without ``anchor_times`` a bare part has none,
    unless it lasts :data:`STATIC_LONG_S`."""
    out: set[Any] = set()
    bare: list[list[dict[str, Any]]] = []
    anchors = sorted(anchor_times or [])
    for part, text in static_parts(frames, feed_times, text_spans or []):
        need = STATIC_MIN_S if text else STATIC_BARE_MIN_S
        if len(part) >= STATIC_MIN_FRAMES and part[-1]["t"] - part[0]["t"] >= need:
            if text:
                out.update(p["id"] for p in part)
            else:
                bare.append(part)
    for part in bare:
        i = bisect.bisect_left(anchors, part[0]["t"] - STATIC_ANCHOR_S)
        near = i < len(anchors) and anchors[i] <= part[-1]["t"] + STATIC_ANCHOR_S
        if near or part[-1]["t"] - part[0]["t"] >= STATIC_LONG_S:
            out.update(p["id"] for p in part)
    return out


def feed_stacks(rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The lines of new menu text (see :data:`TEXT_LINES`) of one feed region's rows that
    stay (spans (first read, end), see :func:`staying_spans`): those that overlap are one
    stack, one line (first read to last end), and with it, of every group of
    :data:`TEXT_TOGETHER` or more rows first read in the same frame, the other rows, each a
    line of its own but the one that ends last. A feed adds its rows one at a time (two at
    most in one frame, a double kill), a screen opens with its lines at once. A stack never
    has more lines than rows on screen."""
    stacks: list[list[tuple[float, float]]] = []
    for a, b in sorted(rows):
        if stacks and a <= max(e for _, e in stacks[-1]):
            stacks[-1].append((a, b))
        else:
            stacks.append([(a, b)])
    out: list[tuple[float, float]] = []
    for stack in stacks:
        out.append((stack[0][0], max(b for _, b in stack)))
        groups: dict[float, list[float]] = {}
        for a, b in stack:
            groups.setdefault(a, []).append(b)
        for a, ends in groups.items():
            if len(ends) >= TEXT_TOGETHER:
                out += [(a, b) for b in sorted(ends, reverse=True)[1:]]
    return out


def staying_spans(
    frames: list[dict[str, Any]],
    text_spans: list[tuple[float, float]],
    text_keys: list[Any] | None = None,
    max_gap: float = STATIC_MAX_GAP_S,
) -> list[tuple[float, float]]:
    """The ``text_spans`` (first read, end) of lines that stay (see :data:`TEXT_LINES`):
    on screen :data:`TEXT_STAY_S` to :data:`TEXT_STAY_MAX_S`, and not of a HUD line,
    whose spans (those with the same ``text_keys`` entry, the line's text and region;
    without keys every span is a line of its own) reach from the read of ``frames`` (in
    time order, within :data:`TEXT_STAY_S` or ``max_gap``) to its end and cover
    :data:`TEXT_HUD_SHARE` of it."""
    if not frames:
        return []
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    allow = max(max_gap, TEXT_STAY_S)
    keys = list(range(len(text_spans))) if text_keys is None else text_keys
    lines: dict[Any, list[tuple[float, float]]] = {}
    for span, key in zip(text_spans, keys, strict=True):
        lines.setdefault(key, []).append(span)
    hud = {
        key
        for key, spans in lines.items()
        if min(a for a, _ in spans) <= t0 + allow
        and max(b for _, b in spans) >= t1 - allow
        and sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in spans) >= TEXT_HUD_SHARE * (t1 - t0)
    }
    return [
        (a, b)
        for (a, b), key in zip(text_spans, keys, strict=True)
        if TEXT_STAY_S <= b - a <= TEXT_STAY_MAX_S and key not in hud
    ]


def staying_lines(
    frames: list[dict[str, Any]],
    text_spans: list[tuple[float, float]],
    text_keys: list[Any] | None = None,
    max_gap: float = STATIC_MAX_GAP_S,
) -> list[tuple[int, int]]:
    """Per frame of ``frames`` (in time order): how many :func:`staying_spans` are on
    screen at its time, and the most of them first read within
    :data:`STATIC_FEED_GUARD_S` of each other."""
    spans = staying_spans(frames, text_spans, text_keys, max_gap)
    starts, ends = sorted(a for a, _ in spans), sorted(b for _, b in spans)
    out = []
    for f in frames:
        t = f["t"]
        shown = sorted(a for a, b in spans if a <= t <= b)
        together = max(
            (bisect.bisect_right(shown, a + STATIC_FEED_GUARD_S) - i for i, a in enumerate(shown)),
            default=0,
        )
        out.append((bisect.bisect_right(starts, t) - bisect.bisect_left(ends, t), together))
    return out


def text_frames(
    frames: list[dict[str, Any]],
    text_spans: list[tuple[float, float]],
    text_keys: list[Any] | None = None,
    max_gap: float = STATIC_MAX_GAP_S,
    feed_times: set[float] | None = None,
) -> set[Any]:
    """Ids of the frames of a text screen (see :data:`TEXT_LINES`): stretches of at
    least :data:`STATIC_MIN_FRAMES` frames over at least :data:`TEXT_MIN_S`, at most
    ``max_gap`` apart, that each show at least :data:`TEXT_LINES` :func:`staying_lines`,
    :data:`TEXT_TOGETHER` of them first read together, and are further than
    :data:`STATIC_FEED_GUARD_S` from every time of ``feed_times`` (feed sightings and kills)."""
    out: set[Any] = set()
    run: list[dict[str, Any]] = []
    busy = sorted(feed_times or ())

    def near_feed(t: float) -> bool:
        i = bisect.bisect_left(busy, t - STATIC_FEED_GUARD_S)
        return i < len(busy) and busy[i] <= t + STATIC_FEED_GUARD_S

    shown = staying_lines(frames, text_spans, text_keys, max_gap)
    for f, (n, together) in [*zip(frames, shown, strict=True), (None, (0, 0))]:
        screen = (
            f is not None
            and n >= TEXT_LINES
            and together >= TEXT_TOGETHER
            and not near_feed(f["t"])
        )
        if f is not None and screen and (not run or f["t"] - run[-1]["t"] <= max_gap):
            run.append(f)
            continue
        if len(run) >= STATIC_MIN_FRAMES and run[-1]["t"] - run[0]["t"] >= TEXT_MIN_S:
            out.update(r["id"] for r in run)
        run = [f] if f is not None and screen else []
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
    ``still`` run id that hold still are menus too (:func:`static_frames`), and so
    are short gaps between menu screens (:func:`_fill_menu_gaps`).
    """
    rules, _ = menu_rules(frames, params, feed_times)
    return {f["t"] for f in frames if rules[f["id"]] != "none"}


def menu_rules(
    frames: list[dict[str, Any]],
    params: DiscoverParams | None = None,
    feed_times: set[float] | None = None,
    text_spans: list[tuple[float, float]] | None = None,
    text_keys: list[Any] | None = None,
    max_gap: float = STATIC_MAX_GAP_S,
) -> tuple[dict[Any, str], dict[str, float]]:
    """The :func:`menu_times` decision per frame id: the rule that made it a menu
    (``cover``, ``jump``, ``list``, ``static``, ``text`` or ``fill``, a gap between
    menu screens, :func:`_fill_menu_gaps`, which runs only with ``feed_times``) or
    ``none``, and the
    source's numbers (``median`` box count, ``min_boxes``, the ``cover``, ``list`` and
    ``jump`` thresholds, inf when off, the ``static_s`` (with new menu text,
    ``text_spans``), ``static_bare_s`` (without, near menu frames) and ``static_diff`` of the static
    rule, which runs only with ``feed_times``, and the ``text_lines`` of the text
    rule (:func:`text_frames`, with ``text_keys``, ``max_gap`` and ``feed_times``), which runs
    only with ``feed_times`` and ``text_spans``)."""
    params = params or DiscoverParams()
    median = float(np.median([len(f["boxes"]) for f in frames])) if frames else 0.0
    lists = median < params.menu_min_boxes / params.menu_jump
    if lists:
        params = replace(params, menu_jump=float("inf"))
    keep = {f["id"] for f in gameplay_frames(frames, params)}
    on = feed_times is not None and text_spans is not None
    texts = text_frames(frames, text_spans or [], text_keys, max_gap, feed_times) if on else set()
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
        elif f["id"] in texts:
            rules[f["id"]] = "text"
        else:
            rules[f["id"]] = "none"
    if feed_times is not None:  # the static rule goes before the text rule
        anchor_times = [f["t"] for f in frames if rules[f["id"]] != "none"]
        static = static_frames(frames, feed_times, text_spans, anchor_times)
        for f in frames:
            if f["id"] in static and rules[f["id"]] in ("none", "text"):
                rules[f["id"]] = "static"
    if feed_times is not None:
        _fill_menu_gaps(frames, rules, feed_times)
    info = {
        "median": median,
        "min_boxes": params.menu_min_boxes,
        "cover": params.menu_cover,
        "list": LIST_MENU_COVER if lists else float("inf"),
        "jump": float("inf") if lists else params.menu_jump * median,
        "static_s": STATIC_MIN_S if feed_times is not None else float("inf"),
        "static_bare_s": STATIC_BARE_MIN_S if feed_times is not None else float("inf"),
        "static_diff": STATIC_MAX_DIFF,
        "static_anchor_s": STATIC_ANCHOR_S if feed_times is not None else float("inf"),
        "static_long_s": STATIC_LONG_S if feed_times is not None else float("inf"),
        "text_lines": TEXT_LINES if on else float("inf"),
    }
    return rules, info


def _fill_menu_gaps(
    frames: list[dict[str, Any]], rules: dict[Any, str], feed_times: set[float]
) -> None:
    """Mark ``fill`` in ``rules`` the frames of ``frames`` (in time order) that no rule made
    a menu between two :data:`FILL_ENDS` frames at most :data:`MENU_FILL_S` apart, when no
    time of ``feed_times`` lies between them (see :data:`MENU_FILL_S`)."""
    busy = sorted(feed_times)
    last = None  # index of the last FILL_ENDS frame
    for i, f in enumerate(frames):
        if rules[f["id"]] not in FILL_ENDS:
            continue
        if last is not None and 0 < f["t"] - frames[last]["t"] <= MENU_FILL_S:
            a, b = frames[last]["t"], f["t"]
            j = bisect.bisect_right(busy, a)
            if not (j < len(busy) and busy[j] < b):
                for g in frames[last + 1 : i]:
                    if rules[g["id"]] == "none":
                        rules[g["id"]] = "fill"
        last = i


def _log_menu_frames(
    name: str,
    frames: list[dict[str, Any]],
    rules: dict[Any, str],
    info: dict[str, float],
    events: list[dict[str, Any]],
    feed_times: set[float] | None = None,
    text_spans: list[tuple[float, float]] | None = None,
    text_keys: list[Any] | None = None,
    max_gap: float = STATIC_MAX_GAP_S,
) -> None:
    """One DEBUG line per frame that is a menu or has an OCR sighting: its numbers and
    the rule that fired, no OCR text (logs get shared). ``menu text`` tells whether
    the frame's still part shows new menu text (``-``: not in such a part);
    ``staying lines`` counts its :func:`staying_lines` and how many of them were first
    read together (text rule)."""
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

    stays = dict.fromkeys((f["id"] for f in frames), (0, 0))
    if text_spans is not None:
        shown = staying_lines(frames, text_spans, text_keys, max_gap)
        stays = dict(zip((f["id"] for f in frames), shown, strict=True))

    def off(v: float, fmt: str) -> str:
        return "off" if v == float("inf") else f">= {v:{fmt}}"

    jump, lists = off(info["jump"], ".0f"), off(info["list"], ".2f")
    texts = off(info["text_lines"], ".0f")
    texts += "" if texts == "off" else f", together >= {TEXT_TOGETHER}"
    static = (
        "off"
        if info["static_s"] == float("inf")
        else f">= {info['static_s']:.1f}s with menu text, >= {info['static_bare_s']:.1f}s within {info['static_anchor_s']:.0f}s of menus or >= {info['static_long_s']:.0f}s without"
    )
    for f in frames:
        rule, n = rules[f["id"]], seen.get(f["t"], 0)
        if rule == "none" and not n:
            continue
        logger.debug(
            "%s menu test %.1fs: %d boxes (min %d), centre cover %.2f (cover >= %.2f, list %s), "
            "median %.1f, jump %s, still %.1fs (static %s), menu text %s, "
            "%d staying lines, %d together (text %s), %d OCR events -> %s",
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
            *stays[f["id"]],
            texts,
            n,
            rule,
        )


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
    ``every``, or that its "mask" gate did not let through, has no record.
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


def readable_regions(
    regions: tuple[Region, ...], width: int, height: int, name: str
) -> tuple[Region, ...]:
    """The ``regions`` with a crop in a ``width`` x ``height`` frame; the others are left
    out with one WARNING each (a rectangle scaled outside the frame has an empty crop)."""
    out = []
    for r in regions:
        _, _, w, h = scaled(r, width, height)
        if w > 0 and h > 0:
            out.append(r)
        else:
            logger.warning(
                "%s: region %s@%d,%d %dx%d has no pixels in the %dx%d frame; skipped",
                name, r.name, r.x, r.y, r.w, r.h, width, height,
            )  # fmt: skip
    return tuple(out)


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
        x, y, w, h = _still_rect(regions, rects)
        self.still = (slice(y - union[1], y - union[1] + h), slice(x - union[0], x - union[0] + w))
        self.last_mask: dict[str, np.ndarray | None] = {r.name: None for r in regions}
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

    def _look(
        self, r: Region, crop: np.ndarray
    ) -> tuple[dict[str, Any] | None, list[tuple[float, float, float, float]] | None, list | None]:
        """The gate of region ``r`` and its OCR calls but the row recognition: ``(record,
        rows, items)``. ``record`` is None when the "mask" gate did not let it through;
        a det-gated region to read has its ``rows`` (``y0, x0, y1, x1``), a mask-gated one
        its ``items``; neither when the layout did not change."""
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
        if r.gate == "read":
            return {}, None, self._full_read(r, crop)
        mask = _small(hud_mask(crop))
        if mask.mean() < r.min_fill:
            self.last_mask[r.name] = mask
            return None, None, None
        prev = self.last_mask[r.name]
        if prev is not None and np.mean(np.abs(mask - prev) > 0.25) < r.min_change:
            return None, None, None
        self.last_mask[r.name] = mask
        return {}, None, self._full_read(r, crop)

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
            if g is None:
                continue
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
        arrays (masks; with the det skip its images)."""
        masks = {
            f"mask{i}": m
            for i, r in enumerate(self.regions)
            if (m := self.last_mask[r.name]) is not None
        }
        state: dict[str, Any] = {"layout": self.last_layout, "rows": self.last_rows}
        if self.det_skip:
            state["boxes"] = self.last_boxes
            masks |= {
                f"anchor{i}": a
                for i, r in enumerate(self.regions)
                if (a := self.anchor.get(r.name)) is not None
            }
        return state, masks

    def restore(self, state: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        """Continue from a :meth:`state`."""
        layout = state["layout"]
        self.last_layout = {
            r.name: None if layout.get(r.name) is None else [tuple(v) for v in layout[r.name]]
            for r in self.regions
        }
        self.last_rows = {k: [tuple(v) for v in rows] for k, rows in state["rows"].items()}
        self.last_mask = {r.name: arrays.get(f"mask{i}") for i, r in enumerate(self.regions)}
        self.last_boxes = {
            k: [list(b) for b in boxes] for k, boxes in state.get("boxes", {}).items()
        }
        self.anchor = {
            r.name: a
            for i, r in enumerate(self.regions)
            if (a := arrays.get(f"anchor{i}")) is not None
        }


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
        self.interp = Interpreter(params, height / 1080, preset)
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
        # static screens (see STATIC_THUMB): only when the HUD area sees enough of the frame
        _, _, still_w, still_h = _still_rect(regions, rects)
        area = still_w * still_h / max(1, width * height)
        self.still_runs = StillRuns(params.fps) if self.menus and area >= STATIC_MIN_AREA else None
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
        self.ref = {n: (x, y, 1920 / width, 1080 / height) for n, (x, y, _, _) in rects.items()}

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
                    rx, ry, sx, sy = self.ref[r.name]
                    self.det_boxes[r.name] = [
                        [
                            _plain((rx + min(p[0] for p in b)) * sx),
                            _plain((ry + min(p[1] for p in b)) * sy),
                            _plain((rx + max(p[0] for p in b)) * sx),
                            _plain((ry + max(p[1] for p in b)) * sy),
                        ]
                        for b in interp.feed.notice_boxes
                    ]
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
        if report:
            stats["feed_regions"] = report

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
                frames began more than :data:`STATIC_FEED_GUARD_S` before the line: gameplay
                can pass as a menu, but a coloured menu row opens with its menu."""
                if not feed_event(e):
                    return False
                t0 = e["_times"][0]
                return e.get("entry") not in LINE_ENTRIES or (
                    _two_hues(e) and any(a + STATIC_FEED_GUARD_S < t0 <= b for a, b in menu_spans)
                )

            # still screens' opening times: menu rows read as icon kills open with the
            # screen, a real kill turns up partway through a still stretch
            opened = [
                run[0]["t"]
                for run in still_stretches(self.frame_boxes)
                if run[0].get("still") is not None and len(run) >= STATIC_MIN_FRAMES
            ]

            def feed_activity(e: dict[str, Any]) -> bool:
                """Feed events that guard still frames from the static rule: not icon kills
                first read within STATIC_FEED_GUARD_S of a still screen's opening."""
                if not feed_event(e):
                    return False
                t0 = e["_times"][0]
                icon = e.get("entry") == "icon"
                return not icon or all(abs(t0 - t) > STATIC_FEED_GUARD_S for t in opened)

            tracked = tracker.events
            feed_times = {t for e in tracked if feed_activity(e) for t in e["_times"]}
            # new menu text: when texts that are no preset phrase were first read, until when
            # shown; ``end`` moves only on the frames a region is read, so it is padded by one
            # read interval
            step = {r.name: r.every / self.interp.params.fps for r in self.regions}
            unknown = [e for e in tracked if e.get("unknown")]

            def shown(e: dict[str, Any]) -> tuple[float, float]:
                return e["_times"][0], e["end"] + step.get(e.get("region"), 0.0)

            max_gap = STATIC_MAX_GAP_S if self.still_runs is None else self.still_runs.max_gap
            frames = self.frame_boxes
            lines = [e for e in unknown if e.get("region") not in feeds]
            text_spans = [shown(e) for e in lines]
            text_keys: list[Any] = [(e.get("region"), e["_key"]) for e in lines]  # a line's spans
            for name in sorted(feeds):  # a feed's rows are one stack (see TEXT_LINES)
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
            if logger.isEnabledFor(logging.DEBUG):
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
        events = collapse_banner_bursts(events)
        events = learn_announcements(events, self.interp.digits.finish())
        if any(r.kind == "score" for r in self.regions):
            scores, stats["score_slots"] = self.scores.finish(menu)
            events += scores
        for e in events:
            e["kind"] = "ocr"
        return events


def _log_score_slots(src_name: str, stats: dict[str, Any]) -> None:
    """One INFO line naming the ``score`` spots of ``stats["score_slots"]`` that were turned
    off as counters, if any (fresh and cached results alike)."""
    off = [k for k, v in stats.get("score_slots", {}).items() if v.get("disabled")]
    if off:
        logger.info(
            "%s: score readout off for %s (a counter, not a score)",
            src_name,
            ", ".join(sorted(off)),
        )


def _raw_region_keys(regions: list[Region]) -> list[dict[str, Any]]:
    """Region keys for :func:`ocr_raw.raw_key`.

    A "score" region is left out of the static-screen thumbnails (:func:`_still_rect`): keyed
    apart, so a read made before with it in them is read again.
    """
    return [r.key() | ({"still": False} if r.kind == "score" else {}) for r in regions]


def detect_ocr_events(
    src: Source,
    params: OcrParams,
    engine: OcrEngine | None,
    force: bool = False,
    debug_dir: Path | None = None,
    avoid: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    make_engine: Callable[[], OcrEngine] | None = None,
    cancel: threading.Event | None = None,
    decode: DecodeHook | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OCR HUD events for one source (cached per source + parameters).

    What menu and scoreboard frames show is dropped (:func:`menu_times`): every frame records the text boxes the "det" gates
    last detected in its regions other than feeds (a region keeps its boxes on
    the frames it is skipped) and the still run of the decoded HUD area without the
    "score" regions (:func:`_still_rect`, :func:`static_frames`; when that area is at least :data:`STATIC_MIN_AREA` of
    the frame), and after the pass the sightings in menu frames
    are forgotten.
    No OCR runs for this; ``stats["menu_frames"]`` counts them (``stats["menu_spans"]``:
    their ``[start, end]`` spans, :func:`~cutpoint.events.menu_spans.merge_menu_frames`), of which
    ``stats["static_frames"]`` held still and ``stats["text_frames"]`` showed many
    lines of new text that stay (:func:`text_frames`).

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
            read (a replay or a cache hit loads no OCR model).
        cancel: For a read in a background thread (:mod:`cutpoint.events.ocr_prefetch`):
            once set, the read stops at the next frame with :class:`OcrCancelled`, its
            ffmpeg killed and nothing committed after the last finished raw chunk. ffmpeg
            then runs isolated from Ctrl+C and is killed at once on ``cancel``. A pass
            that ffmpeg left with an exit the decode did not cause (the console's
            Ctrl+Break reaches every process group) and a failed GPU pass also raise
            :class:`OcrCancelled`, without committing the pass: the caller reads the
            source itself then, resuming from the raw chunks, as a serial run would.
        decode: With ``cancel``: gives the frames of a fresh GPU read of the whole source
            instead of :func:`iter_frames` (the shared decode of
            :mod:`cutpoint.events.ocr_prefetch`), the same frames; asked once at most.

    Returns:
        (events, stats).

    Raises:
        OcrCancelled: With ``cancel``: when it was set, ffmpeg was stopped from outside or a
            GPU pass failed.
    """
    game = game_key_for_path(str(src.path), params.game, params.project)
    preset = load_preset(game, params.project)
    regions = preset.all_regions() if preset else ()
    if not regions:
        if preset is not None:
            logger.info("%s: HUD preset %r has no usable region; skipping OCR", src.name, game)
        else:
            logger.info(
                "%s: no HUD preset for %r; skipping OCR (run with --discover-regions)",
                src.name,
                game,
            )
        return [], {"skipped": "no preset", "game": game}
    avoid = sorted(avoid or [])
    newest = str(preset.feed.get("newest", "bottom")) if preset is not None else "bottom"
    raw = ocr_raw.RawStore(
        src.cdir,
        ocr_raw.raw_key(
            params.fps,
            _raw_region_keys(regions),
            newest,
            MIN_TEXT_SCORE,
            avoid,
            [DET_SKIP_SCALE, DET_SKIP_TILE, DET_SKIP_MAX_DIFF] if params.det_skip else None,
        ),
    )
    key = params.key(game, preset) | ({"avoid": avoid} if avoid else {})
    key["raw"] = cache.params_hash(raw.key)
    cpath = layer_cache_path(src.cdir, "ocr", key)
    if not force and (hit := cache.load_json(cpath)) is not None:
        _log_score_slots(src.name, hit.get("stats", {}))
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
    regions = readable_regions(regions, width, height, src.name)
    if not regions:
        return [], {"skipped": "no region in the frame", "game": game}
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
            if cancel is not None and cancel.is_set():
                raise OcrCancelled
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
        pos: dict[str, Any] = {
            "seg": 0,
            "idx": 0,
            "frame_no": 0,
            "hw": params.hwaccel,
        }
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
            out.mkdir(parents=True, exist_ok=True)
            write_png(out / f"{src.path.stem[-24:]}_{t:07.1f}_{r.name}.png", crop)

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
                exits: list[int] = []  # background: an ffmpeg exit it did not cause
                k = k0
                frames = None
                # "auto" decodes on the CPU when no GPU decoder fits this source: a failed CPU
                # pass is no GPU failure, and a plain retry would only repeat it
                gpu = hw and decoder_for(src.io_path) is not None
                fresh = si == 0 and not k0 and len(segments) == 1 and seg_b >= duration
                if decode is not None and cancel is not None and hw and fresh and not start:
                    frames = decode(union, ocr_vf(params.fps, union), errors, exits)
                    decode = None  # asked once: a retry decodes as usual
                shared = frames is not None  # its split pass failing is retried by a plain decode
                try:
                    for t, frame in frames or iter_frames(
                        src.io_path,
                        params.fps,
                        union,
                        hw,
                        errors,
                        start,
                        None if seg_b >= duration else seg_b - start,
                        cancel=cancel,
                        exit_codes=exits if cancel is not None else None,
                    ):
                        if cancel is not None and cancel.is_set():
                            raise OcrCancelled
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
                    if cancel is not None and (cancel.is_set() or exits):
                        raise OcrCancelled  # killed on cancel, or stopped from outside
                    stats["hwaccel"] = gpu
                    break
                except RuntimeError as err:
                    if cancel is not None and (cancel.is_set() or exits or gpu or shared):
                        # cancelled or stopped from outside, or a failed GPU pass: the read in
                        # its turn redoes it from the raw chunks (CPU retry, hwaccel stats) as a
                        # serial run would
                        raise OcrCancelled from err
                    if k0 and k == k0 and start > seg_b - 1.0:
                        # resumed at the very end of the segment: nothing was left
                        logger.debug("%s: nothing left after %.1fs: %s", src.name, start, err)
                        break
                    if gpu or shared:
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
    if "det_skipped" in stats:
        skipped = sum(stats["det_skipped"].values())
        logger.info(
            "%s: det gate skipped on %d of %d region reads (unchanged since its last det call)",
            src.name,
            skipped,
            skipped + sum(stats["det_calls"].values()),
        )
    _log_score_slots(src.name, stats)
    cache.save_json(
        cpath, {"source": str(src.path), "params": key, "stats": stats, "events": events}
    )
    return events, stats
