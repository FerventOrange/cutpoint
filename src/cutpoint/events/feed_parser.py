"""Kill-feed parsing of the OCR layer (split from ``killfeed.py``, #1085 A347).

:class:`FeedParser` reads kill rows, icon kill lines and notices from a "feed" region's OCR
boxes, with the text and player-tag matching (:func:`fuzzy_contains`, :func:`is_player`,
:func:`player_side`) and the pixel helpers (:func:`text_hue`, :func:`icon_split`) it uses.
"""

from __future__ import annotations

import difflib
import functools
import re
from collections.abc import Callable, Iterable
from itertools import pairwise
from typing import Any

import numpy as np

from cutpoint import state
from cutpoint.events.text import alnum_words, notice_letters

# --------------------------------------------------------------------------- text utils


def norm(text: str) -> str:
    """Upper-case letters and digits in any script and single spaces (for fuzzy comparisons,
    :func:`~cutpoint.events.text.alnum_words`)."""
    return alnum_words(text)


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
    toks = re.findall(r"[^\s\d]+|\d+", t)
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
    """SequenceMatcher ratio of two normalized strings; 0 when both are empty (two texts
    with no letters or digits are no match)."""
    x, y = norm(a).replace(" ", ""), norm(b).replace(" ", "")
    if not (x or y):
        return 0.0
    return difflib.SequenceMatcher(None, x, y).ratio()


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


def by_centre[T](
    items: Iterable[T],
    centre: Callable[[T], Any],
    tol: float,
    order: Callable[[T], Any] | None = None,
) -> list[list[T]]:
    """``items`` grouped into rows, top to bottom (#1085 A338): in ``order`` (by default by
    ``centre``), an item whose ``centre`` is within ``tol`` of the centre of the first item
    of the row before joins that row, else it starts one. Each caller keeps its tolerance."""
    rows: list[list[T]] = []
    anchor: Any = 0.0
    for it in sorted(items, key=order or centre):
        cy = centre(it)
        if rows and abs(anchor - cy) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
            anchor = cy
    return rows


def _rows(items: list[tuple[list, str, float]], tol: float) -> list[list[tuple]]:
    """Group OCR boxes into text rows by vertical centre, each row sorted by x."""
    rows = by_centre(items, lambda i: np.mean([p[1] for p in i[0]]), tol)
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


_CLAN_RE = re.compile(r"^\s*(\[[^\]]{1,8}\]|\^[^\^]{1,8}\^|=[^=]{1,8}=|-[^-]{1,8}-)\s*")
# Generic words that work from the first row on; no game's phrases belong here: a preset's
# ``kill_words`` lists them ("shot down": "kill", "crashed": "crash", ...)
DEFAULT_KILL_WORDS = {
    "destroyed": "kill",
    "killed": "kill",
    "eliminated": "kill",
    "assist": "assist",
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


class FeedParser:
    """Game-agnostic kill-feed parsing for "feed" regions.

    Two forms are recognised (both from one region crop):

    * a row containing a kill word ("A destroyed B", "A eliminated B"): actor before the
      word, target after it;
    * icon feeds without words, where an entry is a coloured *actor* line and
      a differently coloured *target* line, or two coloured names on one row
      (``"entry": "one_line"`` reads only these; any other ``entry``, such as the
      default ``"auto"``, reads both);
      without a colour map, in a discovered icon feed (``"row_notices": true``), a
      row of two names split by an icon gap is a kill, left name first
      (:meth:`_icon_line`).

    A clock-stamped log's wrapped rows are joined first; only the kill words read them, as a
    log tells much more than kills.

    Team colours come from the preset (``"colors": {"red": "enemy", "blue":
    "ally", ...}``, keyed by the colour names :func:`text_hue` returns). A kill
    is the player's when the actor matches one of the player tags (fuzzy,
    ignoring ``_ . -``). Without the preset's own ``kill_words`` the generic
    :data:`DEFAULT_KILL_WORDS` read the rows, and a ``"Name (group)"`` group fills no
    ``killer_vehicle`` or ``victim_vehicle``.
    """

    def __init__(self, cfg: dict[str, Any], tags: tuple[str, ...], scale: float) -> None:
        own = cfg.get("kill_words")
        self.words = {k.upper(): v for k, v in (own or DEFAULT_KILL_WORDS).items()}
        # "Name (Vehicle)": the group is a vehicle only in a feed whose preset words it
        # (its own kill_words); else it is left out of the event's fields
        self.vehicles = bool(own)
        self.colors: dict[str, str] = dict(cfg.get("colors") or {})
        # "one_line" or "auto"; older entry names ("two_line", "paren") read as "auto"
        self.entry = "one_line" if cfg.get("entry") == "one_line" else "auto"
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
        if not self.vehicles:
            a_veh = t_veh = None
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
        if self.entry == "auto":
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
        for k, (line, conf, _, row) in enumerate(lines):
            if k in used:
                continue
            names = _merge_adjacent([i for i in row if len(norm(i[1])) >= 2], gap=14 * self.scale)
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
            if k in skip or len(line.split()) < 2 or not notice_letters(line):
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
