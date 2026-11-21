"""Event tracking of the OCR layer (split from ``killfeed.py``, #1085 A347).

:class:`EventTracker` turns repeated sightings into events; the passes over a source's events
(:func:`merge_kill_reads`, :func:`collapse_banner_bursts`, :func:`learn_announcements`,
:func:`mark_replay_kills`, :func:`link_teamkills`, :func:`derive_streaks`) follow it.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from cutpoint.events.feed_parser import _alnum_len, _fold, norm, similar

# --------------------------------------------------------------------------- tracking


FUZZY = 0.8  # SequenceMatcher ratio for phrase / dedupe matches
PERSIST_S = 8.0  # an entry seen again within this is the same event
# Kill lines tracked by their two names: icon lines and the kills of a clock-stamped log
LINE_ENTRIES = ("icon", "text")
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


_NAME_RUN_RE = re.compile(r"[^\W\d_]{3}")  # three letters in a row, any script


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
    return rest if _NAME_RUN_RE.search(rest) else ""


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
    return line if _NAME_RUN_RE.search(line) else ""


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


def link_teamkills(events: list[dict[str, Any]], generic_teamkills: bool = False) -> None:
    """(Re)compute the ``teamkill`` flag of kill events in place.

    A kill is a teamkill only when ``generic_teamkills`` (the preset's
    ``feed.teamkills``: it trusts its colours) and both name colours read as ally, and never
    one read during a replay (``replay``, :func:`mark_replay_kills`).
    Applied to cached results too, so the rule can change without re-OCR.
    """
    for e in events:
        if e.get("type") == "kill":
            e["teamkill"] = bool(
                generic_teamkills
                and not e.get("replay")
                and e.get("killer_team") == "ally"
                and e.get("victim_team") == "ally"
            )


# Replays (#1069): a "replay" region (the vision HUD map's replay or killcam marker,
# :mod:`cutpoint.events.hud_map`) with text in it, REPLAY_MIN_LETTERS letters at least, in
# text like the region's learned marker (``marker``, :func:`cutpoint.events.regions.marker_seen`:
# a round notice or a name shown in the same place is no replay), read REPLAY_MIN_SIGHTINGS
# times or on screen for REPLAY_MIN_S (the det gate reads a still marker once and keeps it
# while its text boxes stay), is a replay on screen from its first read to its last. A kill
# first read then, or within one read interval after it (never before: a replay often starts
# right after the live kill it shows, which stays the player's), is a kill shown again (a
# killcam, an end-of-round replay): it already happened live, so it is no kill or death of the
# player and no teamkill. The event stays, marked ``replay`` (``replay_of``: the time of the
# live read of the same kill, when there is one), and scores like another player's kill
# (nothing of its own: the player's kills are the only kills that score).
# A replay can go on after its marker is gone (another point of view): a kill read within
# REPLAY_CHAIN_S after a replay span's end, or after the last kill so taken, is in the replay
# too, up to REPLAY_CHAIN_MAX_S after the marker's end (a busy feed does not chain on for
# ever). And a kill read again with the killer and victim of a live kill read before a replay
# started, within REPLAY_COPY_S of it, is that kill shown again (a kill of the player or a
# death of the player alike), when it is read no later after the replay's start than the live
# kill was before it, plus REPLAY_CHAIN_S (a replay runs at about real speed). Both need a replay span: a game without a
# marker found never loses a kill read twice (a respawn game can have the same pair twice).
REPLAY_MIN_LETTERS = 3
REPLAY_MIN_SIGHTINGS = 2
REPLAY_MIN_S = 1.5
REPLAY_CHAIN_S = 8.0
REPLAY_CHAIN_MAX_S = 20.0
REPLAY_COPY_S = 60.0


def replay_spans(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """The ``(start, end)`` of the ``replay`` events read :data:`REPLAY_MIN_SIGHTINGS` times or
    shown for :data:`REPLAY_MIN_S`."""
    return [
        (float(e["t"]), float(e["end"]))
        for e in events
        if e.get("type") == "replay"
        and (
            e.get("sightings", 1) >= REPLAY_MIN_SIGHTINGS
            or float(e["end"]) - float(e["t"]) >= REPLAY_MIN_S
        )
    ]


def mark_replay_kills(
    events: list[dict[str, Any]], spans: list[tuple[float, float]], pad: float
) -> int:
    """Mark the kills shown again in a replay as ``replay`` and clear their player and
    teamkill flags, in place: those first read in a replay span (or ``pad`` seconds after it,
    never before it), those read on within :data:`REPLAY_CHAIN_S` after it (the span grows
    with each, up to :data:`REPLAY_CHAIN_MAX_S` after the marker's end) and copies of a live
    kill read before a span started, within :data:`REPLAY_COPY_S` of it and no later after
    the span's start than the live kill was before it (plus :data:`REPLAY_CHAIN_S`). A copy
    gets the live kill's time as ``replay_of``; the live kill keeps its credit. Returns how
    many."""
    n = 0
    live: list[dict[str, Any]] = []
    # [start, end so far, the marker's own end]
    marks = sorted([float(a), float(b), float(b)] for a, b in spans)
    for e in sorted((e for e in events if e.get("type") == "kill"), key=lambda e: e["t"]):
        t = float(e["t"])
        span = next((s for s in marks if s[0] <= t <= s[1] + pad), None)
        if span is None:
            span = next(
                (
                    s
                    for s in marks
                    if s[1] < t <= min(s[1] + REPLAY_CHAIN_S, s[2] + REPLAY_CHAIN_MAX_S)
                ),
                None,
            )
            if span is not None:
                span[1] = t  # the replay goes on (another point of view)

        def replayed(lt: float, t: float = t) -> bool:
            """A replay started between the live read at ``lt`` and ``t``, and ``t`` is no
            later after its start than ``lt`` was before it (plus REPLAY_CHAIN_S)."""
            return any(lt < s[0] <= t and t - s[0] <= s[0] - lt + REPLAY_CHAIN_S for s in marks)

        same = [
            k
            for k in live
            if t - float(k["t"]) <= REPLAY_COPY_S
            and _names_match(k.get("killer"), e.get("killer"))
            and _names_match(k.get("victim"), e.get("victim"))
            and replayed(float(k["t"]))
        ]
        if span is None and not same:
            live.append(e)
            continue
        if same:
            e["replay_of"] = same[-1]["t"]
        e["replay"] = True
        for k in ("player", "player_kill", "player_death", "teamkill"):
            if k in e:
                e[k] = False
        n += 1
    return n


def _names_match(a: Any, b: Any) -> bool:
    """Whether two OCR reads of a name are the same name (letters and digits, case and a
    misread letter or two aside)."""
    from difflib import SequenceMatcher  # noqa: PLC0415

    x = "".join(ch for ch in str(a or "").lower() if ch.isalnum())
    y = "".join(ch for ch in str(b or "").lower() if ch.isalnum())
    return bool(x and y) and SequenceMatcher(None, x, y, autojunk=False).ratio() >= 0.8


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
                }
            )
        i = j + 1
    events.sort(key=lambda e: e["t"])
