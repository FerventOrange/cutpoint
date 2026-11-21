"""Menu and scoreboard frames of the OCR layer (split from ``killfeed.py``, #1085 A347):
:func:`menu_rules` and the static-screen and staying-text tests it runs (:data:`MENU`).
"""

from __future__ import annotations

import bisect
import heapq
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from cutpoint.events.discover import DiscoverParams, _centre_cover, gameplay_frames
from cutpoint.events.regions import Region

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MenuParams:
    """The thresholds of the menu test (:func:`menu_rules`), one place for all six rules:
    ``cover`` and ``jump`` (``boxes``: discovery's :class:`DiscoverParams` menu fields,
    ``menu_min_boxes``, ``menu_cover`` and ``menu_jump``), ``list``, ``static``, ``text`` and
    ``fill``. The rules read :data:`MENU`. A changed value changes which frames are menus:
    measure it with the menu-time metric of ``scripts/bench.py`` and bump
    :data:`~cutpoint.events.killfeed.GENERIC_VERSION`."""

    boxes: DiscoverParams = field(default_factory=DiscoverParams)  # rules cover and jump
    list_menu_cover: float = (
        0.25  # centre cover of a list menu (2 columns: ~0.33) when the jump is off
    )

    # Static screens (rule "static"): garage, crew and loadout menus show as little text as a busy
    # HUD (6-17 boxes, centre cover 0.04-0.12), so the box rules miss them, but they hold still
    # while gameplay moves. Every OCR frame takes a static_thumb (w, h) grayscale thumbnail of the
    # HUD area it already decoded, "score" regions left out (:func:`_still_rect`, :func:`_thumb_bgr`,
    # :func:`_grey`; no extra decode) and :class:`StillRuns` gives it
    # the id of its still run (``"still"`` in the frame record; no thumbnail is kept). A run goes
    # on while thumbnails stay within static_max_diff of its first frame (mean absolute difference
    # in 0-255 grey levels: still video and its compression noise stay below 1, a moving game view
    # changes by tens of levels, a menu cursor or a small animation by a few) and no frame is
    # missing (a gap over static_max_gap_s, e.g. a skipped bad range, ends it). The frames within
    # static_feed_guard_s of feed activity (a feed sighting or a kill, but not an icon kill first
    # read within static_feed_guard_s of a still run's start: menu rows open with the screen) are
    # taken out, so kills in still gameplay (scoped, spectating) are never dropped; the rest of a run is a static screen
    # where at least static_min_frames frames hold still for at least static_min_s when it shows new
    # menu text, else static_bare_min_s. New menu text is at least static_text_lines texts the
    # generic interpreter reads that are no preset phrase (``unknown`` notice or big text lines, so
    # not numbers or reticle marks), first read in the part or up to static_feed_guard_s before it
    # (a menu shows its words when it opens) and each on screen for at least static_text_share of
    # the part: a menu shows many lines that stay, a gameplay notice is one line or one box that
    # flashes, so a notice read in still gameplay never puts it on the static_min_s path.
    # Still gameplay (scoped, binoculars) holds still for about 6 s, a result screen 6.7 s and menus
    # 17-56 s on the HUD area, so the 5 s minimum alone has no margin; a phrase banner in still
    # gameplay is no menu text, so it never supports dropping itself (menus show phrases too, so
    # phrases cannot guard either). The test runs only when the decoded HUD area is at least
    # static_min_area of the frame: a few small corner regions see too little of the screen (still
    # sky or ground would pass as a menu).
    # Stillness alone is weak evidence: a bare still part (no new menu text) also fits still
    # gameplay (a respawn or spectator wait, a preparation phase), which held still for 11-34 s on
    # hand-labelled footage of two games and was flagged as a menu. A bare part is a static
    # screen only when menu evidence lies within static_anchor_s of it: a frame of another menu
    # rule (``cover``, ``jump``, ``list``, ``text``); a still part with menu text is no evidence
    # (a preparation phase after a still text screen would follow it). Menus come in runs (lobby,
    # garage, matchmaking), so a garage next to its menu screens stays, an isolated still stretch
    # in gameplay does not. A bare part that holds still for static_long_s needs none: the longest
    # still gameplay measured was 37.7 s (a preparation phase), a garage menu held 55.7 s.
    static_thumb: tuple[int, int] = (32, 18)
    static_max_diff: float = 3.0
    static_min_s: float = 5.0
    static_bare_min_s: float = 10.0
    static_min_frames: int = 3
    static_feed_guard_s: float = 2.0
    static_max_gap_s: float = 1.0  # at least; 2 frame intervals at low OCR frame rates
    static_min_area: float = 0.25
    static_text_lines: int = 2
    static_text_share: float = 0.5
    static_long_s: float = 45.0

    # Text screens (rule "text"): lobbies, character select, map screens and result screens rarely
    # hold still (animated backgrounds, cursors, countdowns end their still runs within seconds) and
    # show their words in HUD regions in numbers no box rule sees, but many of their lines stay on
    # screen. A frame is a text screen when at least text_lines lines of new menu text (the
    # ``text_spans`` of :func:`static_parts`, so no preset phrase, number or feed line), each on
    # screen for text_stay_s to text_stay_max_s, are on screen at once, at least text_together of
    # them first read within static_feed_guard_s of each other, in a stretch of at least
    # static_min_frames such frames over at least text_min_s (frames at most static_max_gap_s
    # apart, or 2 frame intervals at low OCR frame rates, as :class:`StillRuns`). A span runs from
    # a line's first read to its last, so it bridges gaps up to PERSIST_S (8 s): a notice that
    # keeps coming back counts as one line that stays. Gameplay text flashes (notices, kill lines,
    # score popups) or is a few lasting labels that arrive one by one (a location, an objective,
    # an overlay, a returning notice); a menu opens with many lines at once. Measured on footage of
    # four games, gameplay showed at most 5 staying lines at once, never for text_min_s and never
    # text_together of them first read together; menus show up to 11. A line that is part of the
    # HUD is no screen that opens and closes: a line on screen for over text_stay_max_s (a player
    # list that stays all match: 7 lines over all 90 s of a long clip), and a line (one text in
    # one region, all its spans: a list hidden by a scoreboard comes back as a new span) first
    # read within text_stay_s of the read's start, on screen until within text_stay_s of its end
    # (a missed read, a region read every few frames or a clip that opens on a fade) and for at
    # least text_hud_share of it, so a source shorter than text_stay_max_s is covered too.
    # Like the static rule it runs only with ``feed_times`` and ``text_spans`` (the decoded HUD
    # area is at least static_min_area of the frame), and like it leaves out the frames within
    # static_feed_guard_s of feed activity (#858): a busy gameplay moment (name tags, location labels
    # and prompts piling up over seconds, the notices of a death) can reach text_lines staying lines,
    # text_together of them first read together, for text_min_s, but the kill feed goes on beside it;
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
    # it in one frame: of every group of text_together or more rows first read in the same frame,
    # the rows count each.
    # Measured on footage of two games (four clips): this cleared a 12 s and a 7 s stretch of
    # gameplay beside a busy feed, kept a lobby whose five lines sit in the feed's area (merged into
    # one, it lost 11 s of menu) and every other menu span, and lost no event.
    text_lines: int = 5
    text_together: int = 3
    text_hud_share: float = 0.5
    text_stay_s: float = 6.0
    text_stay_max_s: float = 60.0
    text_min_s: float = 3.0

    # Gaps between menu screens (rule "fill", #409): loading, matchmaking and "processing" screens
    # between a lobby and the next screen show no text the box or text rules see, and an animated
    # garage menu holds no still run, so a lobby-heavy source kept them as gameplay. Frames that no
    # rule flags between two frames of the box or text rules (``cover``, ``jump``, ``list``,
    # ``text``) at most menu_fill_s apart, with no feed activity between them, are menus too. A
    # static frame is no end of a gap: still gameplay can pass the static rule, and a gap filled
    # from it would grow into the gameplay next to it. Measured on footage of two games
    # (hand-labelled menu time), recall rose from 0.75 to 0.89 and 0.54 to 0.64 with no new
    # gameplay frame flagged; 15 s spans their loading screens, and longer gaps added none. Like
    # the static rule it runs only with ``feed_times``.
    menu_fill_s: float = 15.0
    fill_ends: tuple[str, ...] = ("cover", "jump", "list", "text")

    @property
    def static_anchor_s(self) -> float:
        """Menu evidence for a bare still screen lies this close (see ``static_thumb``):
        ``menu_fill_s``."""
        return self.menu_fill_s


MENU = MenuParams()


def _thumb_bgr(frame: np.ndarray) -> np.ndarray:
    """The :attr:`MenuParams.static_thumb` BGR thumbnail (uint8) of a decoded BGR frame, as the raw
    OCR cache keeps it."""
    import cv2  # noqa: PLC0415

    return cv2.resize(frame, MENU.static_thumb, interpolation=cv2.INTER_AREA)


def _still_rect(
    regions: tuple[Region, ...], rects: dict[str, tuple[int, int, int, int]]
) -> tuple[int, int, int, int]:
    """The HUD area of the static-screen test (:attr:`MenuParams.static_thumb`, :attr:`MenuParams.static_min_area`)
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
    first thumbnail and the last time are kept (see :attr:`MenuParams.static_thumb`)."""

    def __init__(self, fps: float) -> None:
        self.max_gap = max(MENU.static_max_gap_s, 2 / fps) if fps > 0 else MENU.static_max_gap_s
        self.anchor: np.ndarray | None = None
        self.last_t: float | None = None
        self.run = -1

    def add(self, t: float, thumb: np.ndarray) -> int:
        """The run id of the frame at ``t`` with ``thumb``."""
        if (
            self.anchor is None
            or self.last_t is None
            or t - self.last_t > self.max_gap
            or float(np.mean(np.abs(thumb - self.anchor))) > MENU.static_max_diff
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
    """The parts of still runs that may be static screens (see :attr:`MenuParams.static_thumb`):
    frames of a still run further than :attr:`MenuParams.static_feed_guard_s` from every time
    of ``feed_times`` (feed sightings and kills), each with whether it shows new menu
    text: at least :attr:`MenuParams.static_text_lines` of the ``text_spans`` (first read, end)
    first read in the part or up to :attr:`MenuParams.static_feed_guard_s` before it, each
    overlapping at least :attr:`MenuParams.static_text_share` of the part."""
    busy, spans = sorted(feed_times), sorted(text_spans)
    starts = [s for s, _ in spans]

    def any_in(times: list[float], a: float, b: float) -> bool:
        i = bisect.bisect_left(times, a)
        return i < len(times) and times[i] <= b

    def menu_text(a: float, b: float) -> bool:
        lo = bisect.bisect_left(starts, a - MENU.static_feed_guard_s)
        hi = bisect.bisect_right(starts, b)
        stay = sum(
            min(e, b) - max(s, a) >= MENU.static_text_share * (b - a) for s, e in spans[lo:hi]
        )
        return stay >= MENU.static_text_lines

    out: list[tuple[list[dict[str, Any]], bool]] = []
    for run in still_stretches(frames):
        part: list[dict[str, Any]] = []
        for f in [*run, None]:  # None closes the last part
            if (
                f is not None
                and f.get("still") is not None
                and not any_in(
                    busy, f["t"] - MENU.static_feed_guard_s, f["t"] + MENU.static_feed_guard_s
                )
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
    least :attr:`MenuParams.static_min_frames` frames that holds still for at least
    :attr:`MenuParams.static_min_s` when it shows new menu text (``text_spans``), else
    :attr:`MenuParams.static_bare_min_s` and a time of ``anchor_times`` (frames another menu rule
    flagged) within :attr:`MenuParams.static_anchor_s`. Without ``anchor_times`` a bare part has none,
    unless it lasts :attr:`MenuParams.static_long_s`."""
    out: set[Any] = set()
    bare: list[list[dict[str, Any]]] = []
    anchors = sorted(anchor_times or [])
    for part, text in static_parts(frames, feed_times, text_spans or []):
        need = MENU.static_min_s if text else MENU.static_bare_min_s
        if len(part) >= MENU.static_min_frames and part[-1]["t"] - part[0]["t"] >= need:
            if text:
                out.update(p["id"] for p in part)
            else:
                bare.append(part)
    for part in bare:
        i = bisect.bisect_left(anchors, part[0]["t"] - MENU.static_anchor_s)
        near = i < len(anchors) and anchors[i] <= part[-1]["t"] + MENU.static_anchor_s
        if near or part[-1]["t"] - part[0]["t"] >= MENU.static_long_s:
            out.update(p["id"] for p in part)
    return out


def feed_stacks(rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The lines of new menu text (see :attr:`MenuParams.text_lines`) of one feed region's rows that
    stay (spans (first read, end), see :func:`staying_spans`): those that overlap are one
    stack, one line (first read to last end), and with it, of every group of
    :attr:`MenuParams.text_together` or more rows first read in the same frame, the other rows, each a
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
            if len(ends) >= MENU.text_together:
                out += [(a, b) for b in sorted(ends, reverse=True)[1:]]
    return out


def staying_spans(
    frames: list[dict[str, Any]],
    text_spans: list[tuple[float, float]],
    text_keys: list[Any] | None = None,
    max_gap: float = MENU.static_max_gap_s,
) -> list[tuple[float, float]]:
    """The ``text_spans`` (first read, end) of lines that stay (see :attr:`MenuParams.text_lines`):
    on screen :attr:`MenuParams.text_stay_s` to :attr:`MenuParams.text_stay_max_s`, and not of a HUD line,
    whose spans (those with the same ``text_keys`` entry, the line's text and region;
    without keys every span is a line of its own) reach from the read of ``frames`` (in
    time order, within :attr:`MenuParams.text_stay_s` or ``max_gap``) to its end and cover
    :attr:`MenuParams.text_hud_share` of it."""
    if not frames:
        return []
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    allow = max(max_gap, MENU.text_stay_s)
    keys = list(range(len(text_spans))) if text_keys is None else text_keys
    lines: dict[Any, list[tuple[float, float]]] = {}
    for span, key in zip(text_spans, keys, strict=True):
        lines.setdefault(key, []).append(span)
    hud = {
        key
        for key, spans in lines.items()
        if min(a for a, _ in spans) <= t0 + allow
        and max(b for _, b in spans) >= t1 - allow
        and sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in spans)
        >= MENU.text_hud_share * (t1 - t0)
    }
    return [
        (a, b)
        for (a, b), key in zip(text_spans, keys, strict=True)
        if MENU.text_stay_s <= b - a <= MENU.text_stay_max_s and key not in hud
    ]


def staying_lines(
    frames: list[dict[str, Any]],
    text_spans: list[tuple[float, float]],
    text_keys: list[Any] | None = None,
    max_gap: float = MENU.static_max_gap_s,
) -> list[tuple[int, int]]:
    """Per frame of ``frames`` (in time order): how many :func:`staying_spans` are on
    screen at its time, and the most of them first read within
    :attr:`MenuParams.static_feed_guard_s` of each other."""
    spans = sorted(staying_spans(frames, text_spans, text_keys, max_gap))
    starts, ends = [a for a, _ in spans], sorted(b for _, b in spans)
    out: list[tuple[int, int]] = [(0, 0)] * len(frames)
    # one sweep over the frames in time order (#1085 A704): ``shown`` holds the starts of
    # the spans on screen (start <= t <= end), sorted; ``on`` their (end, start)
    shown: list[float] = []
    on: list[tuple[float, float]] = []
    nxt = 0
    for k in sorted(range(len(frames)), key=lambda k: frames[k]["t"]):
        t = frames[k]["t"]
        while nxt < len(spans) and spans[nxt][0] <= t:
            a, b = spans[nxt]
            bisect.insort(shown, a)
            heapq.heappush(on, (b, a))
            nxt += 1
        while on and on[0][0] < t:
            _, a = heapq.heappop(on)
            del shown[bisect.bisect_left(shown, a)]
        together = max(
            (
                bisect.bisect_right(shown, a + MENU.static_feed_guard_s) - i
                for i, a in enumerate(shown)
            ),
            default=0,
        )
        out[k] = (bisect.bisect_right(starts, t) - bisect.bisect_left(ends, t), together)
    return out


def text_frames(
    frames: list[dict[str, Any]],
    text_spans: list[tuple[float, float]],
    text_keys: list[Any] | None = None,
    max_gap: float = MENU.static_max_gap_s,
    feed_times: set[float] | None = None,
) -> set[Any]:
    """Ids of the frames of a text screen (see :attr:`MenuParams.text_lines`): stretches of at
    least :attr:`MenuParams.static_min_frames` frames over at least :attr:`MenuParams.text_min_s`, at most
    ``max_gap`` apart, that each show at least :attr:`MenuParams.text_lines` :func:`staying_lines`,
    :attr:`MenuParams.text_together` of them first read together, and are further than
    :attr:`MenuParams.static_feed_guard_s` from every time of ``feed_times`` (feed sightings and kills)."""
    out: set[Any] = set()
    run: list[dict[str, Any]] = []
    busy = sorted(feed_times or ())

    def near_feed(t: float) -> bool:
        i = bisect.bisect_left(busy, t - MENU.static_feed_guard_s)
        return i < len(busy) and busy[i] <= t + MENU.static_feed_guard_s

    shown = staying_lines(frames, text_spans, text_keys, max_gap)
    for f, (n, together) in [*zip(frames, shown, strict=True), (None, (0, 0))]:
        screen = (
            f is not None
            and n >= MENU.text_lines
            and together >= MENU.text_together
            and not near_feed(f["t"])
        )
        if f is not None and screen and (not run or f["t"] - run[-1]["t"] <= max_gap):
            run.append(f)
            continue
        if len(run) >= MENU.static_min_frames and run[-1]["t"] - run[0]["t"] >= MENU.text_min_s:
            out.update(r["id"] for r in run)
        run = [f] if f is not None and screen else []
    return out


def menu_rules(
    frames: list[dict[str, Any]],
    feed_times: set[float] | None = None,
    text_spans: list[tuple[float, float]] | None = None,
    text_keys: list[Any] | None = None,
    max_gap: float = MENU.static_max_gap_s,
) -> tuple[dict[Any, str], dict[str, float]]:
    """Which of ``frames`` of one source are menu and scoreboard frames.

    ``frames`` are ``{"id", "t", "source", "boxes"}`` records of the text boxes
    the OCR layer detected per frame (reference pixels); the test is
    discovery's :func:`~cutpoint.events.discover.gameplay_frames` (default
    params), with the source's median box count taken over these frames. The
    region boxes of a frame are few (a median of 0 to 6), so the box-count
    jump applies only when the median is at least ``menu_min_boxes /
    menu_jump``. Below that, a frame with ``menu_min_boxes`` boxes is a menu
    when they cover the screen centre (``menu_cover``) or at least
    :attr:`MenuParams.list_menu_cover` of it (a 1- or 2-column list or scoreboard).

    Returns the decision per frame id: the rule that made it a menu
    (``cover``, ``jump``, ``list``, ``static``, ``text`` or ``fill``, a gap between
    menu screens, :func:`_fill_menu_gaps`, which runs only with ``feed_times``) or
    ``none``, and the
    source's numbers (``median`` box count, ``min_boxes``, the ``cover``, ``list`` and
    ``jump`` thresholds, inf when off, the ``static_s`` (with new menu text,
    ``text_spans``), ``static_bare_s`` (without, near menu frames) and ``static_diff`` of the static
    rule, which runs only with ``feed_times``, and the ``text_lines`` of the text
    rule (:func:`text_frames`, with ``text_keys``, ``max_gap`` and ``feed_times``), which runs
    only with ``feed_times`` and ``text_spans``)."""
    params = MENU.boxes
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
            and _centre_cover(f) >= MENU.list_menu_cover
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
        "list": MENU.list_menu_cover if lists else float("inf"),
        "jump": float("inf") if lists else params.menu_jump * median,
        "static_s": MENU.static_min_s if feed_times is not None else float("inf"),
        "static_bare_s": MENU.static_bare_min_s if feed_times is not None else float("inf"),
        "static_diff": MENU.static_max_diff,
        "static_anchor_s": MENU.static_anchor_s if feed_times is not None else float("inf"),
        "static_long_s": MENU.static_long_s if feed_times is not None else float("inf"),
        "text_lines": MENU.text_lines if on else float("inf"),
    }
    return rules, info


def _fill_menu_gaps(
    frames: list[dict[str, Any]], rules: dict[Any, str], feed_times: set[float]
) -> None:
    """Mark ``fill`` in ``rules`` the frames of ``frames`` (in time order) that no rule made
    a menu between two :attr:`MenuParams.fill_ends` frames at most :attr:`MenuParams.menu_fill_s` apart, when no
    time of ``feed_times`` lies between them (see :attr:`MenuParams.menu_fill_s`)."""
    busy = sorted(feed_times)
    last = None  # index of the last MENU.fill_ends frame
    for i, f in enumerate(frames):
        if rules[f["id"]] not in MENU.fill_ends:
            continue
        if last is not None and 0 < f["t"] - frames[last]["t"] <= MENU.menu_fill_s:
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
    max_gap: float = MENU.static_max_gap_s,
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
    texts += "" if texts == "off" else f", together >= {MENU.text_together}"
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


def _union(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """The decoded rectangle: the bounding box of the region ``rects`` (even size)."""
    x0 = min(x for x, _, _, _ in rects)
    y0 = min(y for _, y, _, _ in rects)
    x1 = max(x + w for x, _, w, _ in rects)
    y1 = max(y + h for _, y, _, h in rects)
    return x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
