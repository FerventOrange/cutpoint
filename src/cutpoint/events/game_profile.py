"""Game profiles: a discovered HUD preset enriched with what its text means.

Region discovery (:mod:`cutpoint.events.discover`) finds *where* HUD text is.
:func:`build_profile` adds what can be read from the same discovery OCR
(``discovery/<game>/ocr.jsonl``), with no model, no network and no new
dependency (the ``ocr`` backend):

* ``candidate_phrases``: recurring multi-word texts of the banner regions (and
  the generic big-text region) of the gameplay frames
  (:func:`~cutpoint.events.discover.gameplay_frames`), kept only when they
  fuzzy-occur in at least :data:`MIN_BURSTS` distinct bursts. Texts are split
  into word runs first (:func:`_segments`): words OCR joined are split at
  camelCase boundaries ("blockedShieldSaver" is "blocked Shield Saver"), and
  numbers ("ROUND5", "x1", "2/3", "3RD"), timestamps, gamertags (other words
  mixing letters and digits), the user's player tags, handles, punctuation and
  a chat message after "name: " end a run. A candidate is a run, or a part of
  a run, of 2 to :data:`MAX_WORDS` words, so a phrase inside a log line is
  found. A part is tried only where it also shows without the rest of its
  run. No candidate (a whole run or a part) is tried when its first or last
  word is a slot, as with the names in
  "<name> blocked <name>" lines: the rest of the phrase (2+ words) has at
  least :data:`SLOT_WORDS` different words on that side and the phrase is in
  less than half of its frames, or the edge word itself varies (it also shows
  on the other side of its neighbour, or next to :data:`SLOT_WORDS` other
  words) and the same holds for that neighbour. A family of banners sharing a
  word ("GATE SECURED", "HILL SECURED") is no slot. A longer phrase hides a
  shorter one it contains unless the shorter one occurs in more bursts. Case
  is ignored for counting and matching, as the OCR layer does; a candidate must have one casing in at least
  :data:`CASE_SHARE` of the bursts it shows in and keeps that casing ("Shield
  Saver"). Runs longer than :data:`MAX_WORDS` words (hints, chat), static
  labels (fuzzy-found in most gameplay frames) and phrases close to one of
  :data:`~cutpoint.events.regions.GENERIC_BANNERS` are left out; at most
  :data:`MAX_PHRASES`.
* ``title`` and ``match`` from the game key. ``title`` is the name the VLM
  prompt already derives from the key; ``match`` is left empty unless the key
  is a plausible game name (see :func:`match_hint`).

Regions are kept exactly as discovery proposed them, and ``feed`` colours,
``view`` and ``hype`` are not set: the OCR text alone does not say what they
should be. The result keeps ``"source": "discovered"`` and records its
provenance as ``"profile": {"backend": "ocr", "version": 3}``.

Cache keys: ``source``, ``profile``, ``title``, ``match``, ``notes`` and
``candidate_phrases`` are not part of the OCR cache key
(:meth:`~cutpoint.events.killfeed.OcrParams.key` uses the regions and
:meth:`~cutpoint.events.regions.HudPreset.key`; ``preset_from_dict`` ignores the
last two), and ``title`` equals the derived VLM prompt name. So a profile has
the OCR and VLM cache keys of the plain discovered preset; promoting a phrase
into ``banner_phrases`` changes recognition (banner events) and therefore the
OCR key.

The builder is a pure function of its inputs: the same proposal and OCR
samples give the same profile.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any

from cutpoint.events.discover import DiscoverParams, boxes_in, gameplay_frames
from cutpoint.events.killfeed import fuzzy_contains, is_player, norm, similar
from cutpoint.events.regions import GENERIC_BANNERS, bigtext_region

BACKEND = "ocr"
VERSION = 3  # bump when the OCR backend's output changes
MAX_PHRASES = 12
MIN_BURSTS = 2  # a phrase must be seen in this many distinct bursts
STATIC_SHARE = 0.5  # texts in at least this share of gameplay frames are static labels
FUZZY = 0.8  # same threshold as the OCR layer's phrase matching
MAX_WORDS = 5  # longer texts are sentences (hints, chat), not banners
MAX_CANDIDATES = 100  # most frequent candidates compared fuzzily (bounds the cost)
CASE_SHARE = 2 / 3  # a candidate's casing in at least this share of its bursts
SLOT_WORDS = 4  # an edge word with this many alternatives next to the rest is a slot
NOTES = (
    "candidate_phrases were mined from the discovery OCR and are inactive. To activate one,"
    " move it into banner_phrases."
)
# a handle ("@name", "#tag") or a separator: ends a word run
_BREAK = re.compile(r"[@#]\S*|[:;,!?|/\\()\[\]{}<>\"]")
_CHAT = re.compile(r":\s.*", re.DOTALL)  # "name: message": the message is chat
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")  # "blockedShield" -> "blocked", "Shield"
# a number joined to a word: one letter run and one digit run ("ROUND5", "5KILLS")
_NUMBERED = re.compile(r"([A-Za-z]+)\d+|\d+([A-Za-z]+)")
_ORDINAL = re.compile(r"\d+(?:ST|ND|RD|TH)")  # "3RD", dropped whole

# Words that name a folder of recordings, not a game: a key containing one is
# no safe path hint.
GENERIC_WORDS = frozenset(
    {
        "camera",
        "capture",
        "captures",
        "clip",
        "clips",
        "dcim",
        "desktop",
        "documents",
        "downloads",
        "edits",
        "export",
        "exports",
        "folder",
        "footage",
        "game",
        "gameplay",
        "games",
        "highlights",
        "home",
        "medal",
        "movies",
        "new",
        "nvidia",
        "obs",
        "onedrive",
        "outplayed",
        "output",
        "pictures",
        "projects",
        "raw",
        "recording",
        "recordings",
        "renders",
        "replay",
        "replays",
        "screen",
        "shadowplay",
        "stream",
        "streams",
        "temp",
        "tmp",
        "untitled",
        "users",
        "video",
        "videos",
        "vod",
        "vods",
    }
)


def build_profile(
    proposal: dict[str, Any],
    ocr_samples: list[dict[str, Any]],
    game: str,
    *,
    player_tags: Sequence[str] = (),
) -> dict[str, Any]:
    """Enrich a discovery proposal into a game profile (OCR backend).

    Args:
        proposal: The preset dict :func:`~cutpoint.events.discover.discover_regions`
            returns (or its ``proposal.json``).
        ocr_samples: The discovery OCR records
            (:func:`~cutpoint.events.discover.read_ocr_samples` of ``ocr.jsonl``).
        game: Preset key of the game.
        player_tags: The user's in-game names; words matching one (fuzzy, as
            in the OCR layer) are cut out of the texts, so they end a phrase.

    Returns:
        A preset dict for :func:`~cutpoint.events.regions.preset_from_dict`: the
        proposal plus ``title``, ``match``, mined ``candidate_phrases``,
        ``notes`` and ``profile``; ``banner_phrases`` is the proposal's. The
        inputs are not modified.
    """
    out = copy.deepcopy(proposal)
    out["game"] = game
    out["title"] = game.replace("-", " ").title()
    hint = match_hint(game)
    out["match"] = [hint] if hint else []
    out["candidate_phrases"] = mine_phrases(out, ocr_samples, player_tags)
    out["notes"] = NOTES
    out["profile"] = {"backend": BACKEND, "version": VERSION}
    return out


def match_hint(game: str) -> str:
    """A lower-case path substring for ``game`` ("some-game" -> "some game"), or "".

    Empty when the key has a generic word ("footage", "videos", ...; a letter
    run of a word counts, so "footage8" and "videos_2024" are generic too), has
    no letters, is shorter than 4 characters or is one word shorter than 6,
    since it would select the preset for unrelated folders ("rust" in
    "frustration").
    """
    words = [w for w in game.lower().split("-") if w]
    hint = " ".join(words)
    if (
        len(hint) < 4
        or (len(words) == 1 and len(hint) < 6)
        or not any(c.isalpha() for c in hint)
        or any(r in GENERIC_WORDS for w in words for r in re.findall(r"[a-z]+", w))
    ):
        return ""
    return hint


def mine_phrases(
    preset: dict[str, Any], ocr_samples: list[dict[str, Any]], player_tags: Sequence[str] = ()
) -> list[str]:
    """Recurring banner phrases in the OCR text of ``preset``'s banner regions.

    Only gameplay frames count (:func:`~cutpoint.events.discover.gameplay_frames`
    with the proposal's discovery params). Candidates are word runs of
    :func:`_segments` and their parts of 2 to :data:`MAX_WORDS` words; a part
    that never shows without a longer candidate around it (same frames) is not
    tried, nor is any candidate whose first or last word is a varying slot ("<name>
    blocked"; see the module docstring: :data:`SLOT_WORDS` different words next
    to the rest, or next to the neighbour of a varying edge word, which has the
    candidate in less than half of its frames). Candidates are compared
    case-folded and need one casing in :data:`CASE_SHARE` of the bursts they
    show in (see :func:`_casing`).

    Occurrences are fuzzy (:func:`~cutpoint.events.killfeed.fuzzy_contains`) for
    the burst count and for the static-label share alike; each distinct text is
    compared with a candidate at most once, and only the :data:`MAX_CANDIDATES`
    most frequent candidates are compared at all. Bursts count only raw texts
    OCR would match once the phrase is promoted (plain
    :func:`~cutpoint.events.killfeed.norm`, digits kept: "WON ROUND5" counts,
    "WON 10 ROUND" does not, "blockedShieldSaver:x1" counts for "Shield
    Saver"); the static-label share uses the texts with numbers and tags left
    out, so labels with changing numbers stay static. A kept phrase inside a
    longer kept one is dropped unless it occurs in more bursts.

    Sorted by the number of bursts they occur in (most first), then by text.
    """
    tags = tuple(player_tags)
    known = {f.name for f in fields(DiscoverParams)}
    raw_params = preset.get("discovery", {}).get("params", {})
    params = DiscoverParams(**{k: v for k, v in raw_params.items() if k in known})
    run_ids = preset.get("discovery", {}).get("frames")
    if run_ids is not None:  # the frames of the discovery run (ocr.jsonl only grows)
        ids = set(map(str, run_ids))
        ocr_samples = [f for f in ocr_samples if str(f.get("id")) in ids]
    frames = gameplay_frames(ocr_samples, params)
    rects = [
        (r["x"], r["y"], r["w"], r["h"], int(r.get("min_text_h", 0)))
        for r in preset.get("regions", [])
        if r.get("kind") in ("banner", "bigtext")
    ]
    if preset.get("bigtext", True) and not any(
        r.get("kind") == "bigtext" for r in preset.get("regions", [])
    ):
        big = bigtext_region()
        rects.append((big.x, big.y, big.w, big.h, big.min_text_h))
    # split texts per frame, norm texts per burst; exact frame counts of candidates
    by_frame: dict[str, set[str]] = {}
    by_burst: dict[tuple[str, int], set[str]] = {}
    frames_of: dict[str, set[str]] = {}
    longer: dict[str, set[str]] = {}  # candidates one word longer around each
    span_frames: dict[str, set[str]] = {}  # frames of every span, single words too
    left: dict[str, set[str]] = {}  # distinct words directly left of each span
    right: dict[str, set[str]] = {}  # ... and right of it
    casings: dict[str, dict[str, set[tuple[str, int]]]] = {}  # casing -> bursts
    for f in frames:
        fid, bkey = str(f["id"]), (str(f["source"]), int(f["burst"]))
        for text in {b["text"] for rect in rects for b in boxes_in(f, *rect)}:
            if raw := norm(text):
                by_burst.setdefault(bkey, set()).add(raw)
            runs = _segments(text, tags)
            if split := " ".join(w for run in runs for w in run).upper():
                by_frame.setdefault(fid, set()).add(split)
            for run in runs:
                if len(run) > MAX_WORDS:
                    continue  # a sentence (hint, chat), not a banner
                up = [w.upper() for w in run]
                for i in range(len(run)):
                    for j in range(i + 1, len(run) + 1):
                        sp = " ".join(up[i:j])
                        span_frames.setdefault(sp, set()).add(fid)
                        if i > 0:
                            left.setdefault(sp, set()).add(up[i - 1])
                        if j < len(run):
                            right.setdefault(sp, set()).add(up[j])
                for i in range(len(run) - 1):
                    for j in range(i + 2, len(run) + 1):
                        c = " ".join(run[i:j]).upper()
                        frames_of.setdefault(c, set()).add(fid)
                        casings.setdefault(c, {}).setdefault(" ".join(run[i:j]), set()).add(bkey)
                        around = longer.setdefault(c, set())
                        if i > 0:
                            around.add(" ".join(run[i - 1 : j]).upper())
                        if j < len(run):
                            around.add(" ".join(run[i : j + 1]).upper())
    static = STATIC_SHARE * max(1, len(frames))

    def slot(c: str) -> bool:
        """Whether the first or last word of ``c`` is a varying slot (a name).

        Either the rest of ``c`` (2+ words) has :data:`SLOT_WORDS` different
        words on that side and ``c`` is in less than half of its frames
        ("<name> blocked Shield Saver"), or the edge word itself varies and the
        same holds for its single neighbour ("<name> tagged <name>"). An edge
        word varies when it also shows on the other side of that neighbour or
        next to :data:`SLOT_WORDS` other words; a banner family sharing a word
        ("GATE SECURED", "HILL SECURED", ...) has fixed edge words and stays.
        """
        words = c.split()
        n = len(frames_of[c])
        checks = (
            (words[0], " ".join(words[1:]), words[1], left, right),
            (words[-1], " ".join(words[:-1]), words[-2], right, left),
        )
        for edge, rest, nb, side, other in checks:
            # a constant phrase of 2+ words after/before varying words
            if (
                len(words) >= 3
                and len(side.get(rest, ())) >= SLOT_WORDS
                and n < len(span_frames[rest]) / 2
            ):
                return True
            # the single neighbour only when the edge word itself varies
            free = edge in other.get(nb, ()) or len(other.get(edge, ())) >= SLOT_WORDS
            if free and len(side.get(nb, ())) >= SLOT_WORDS and n < len(span_frames[nb]) / 2:
                return True
        return False

    generic = [norm(g) for g in GENERIC_BANNERS]
    cands = sorted(
        (
            c
            for c, fs in frames_of.items()
            if len(fs) < static  # exact count: cheap pre-filter
            and all(len(frames_of[e]) < len(fs) for e in longer[c])  # shows on its own
            and not slot(c)
        ),
        key=lambda c: (-len(frames_of[c]), c),
    )
    frame_sets = [by_frame[k] for k in sorted(by_frame)]
    burst_sets = [by_burst[k] for k in sorted(by_burst)]
    grams = {t: _bigrams(t) for texts in (*burst_sets, *frame_sets) for t in texts}
    kept: list[tuple[str, int]] = []
    names: dict[str, str] = {}
    tried = 0
    for c in cands:
        if tried >= MAX_CANDIDATES:
            break
        if any(similar(c, k) >= FUZZY for k, _ in kept):
            continue  # an OCR variant of a more frequent phrase
        if any(similar(c, g) >= FUZZY for g in generic):
            continue  # already matched by every JSON preset
        name = _casing(casings[c])
        if name is None:
            continue  # casing varies: not one HUD text
        tried += 1
        hits: dict[str, bool] = {}
        cg = _bigrams(c)

        def occurs(
            texts: set[str], c: str = c, cg: set[str] = cg, hits: dict[str, bool] = hits
        ) -> bool:
            for t in sorted(texts):
                if t not in hits:
                    # a fuzzy match shares a bigram with a phrase of 3+ characters: with
                    # only 1-character matching blocks, M <= (2n+1)/3 < 0.8n
                    hits[t] = bool(cg & grams[t] or len(cg) < 2) and fuzzy_contains(t, c, FUZZY)
                if hits[t]:
                    return True
            return False

        bursts = _count(burst_sets, occurs, len(burst_sets))
        if bursts >= MIN_BURSTS and _count(frame_sets, occurs, static) < static:
            kept.append((c, bursts))
            names[c] = name
    kept = [
        (c, n)
        for c, n in kept
        if not any(k != c and f" {c} " in f" {k} " and kn >= n for k, kn in kept)
    ]
    kept.sort(key=lambda p: (-p[1], p[0]))
    return [names[c] for c, _ in kept[:MAX_PHRASES]]


def _bigrams(text: str) -> set[str]:
    """Character bigrams of ``text`` with spaces removed."""
    t = text.replace(" ", "")
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _count(sets: list[set[str]], occurs: Callable[[set[str]], bool], stop: float) -> int:
    """How many of ``sets`` contain an occurrence (counting stops at ``stop``)."""
    n = 0
    for texts in sets:
        if n >= stop:
            break
        n += occurs(texts)
    return n


def _casing(forms: dict[str, set[tuple[str, int]]]) -> str | None:
    """The casing of a candidate seen in the most bursts (then the first by text),
    or None when it has less than :data:`CASE_SHARE` of the bursts the candidate
    shows in ("Shield Saver" in 2 bursts, "SHIELD saver" in 1: "Shield Saver")."""
    best = min(forms, key=lambda s: (-len(forms[s]), s))
    bursts = set().union(*forms.values())
    return best if len(forms[best]) >= CASE_SHARE * len(bursts) else None


def _segments(text: str, tags: tuple[str, ...] = ()) -> list[list[str]]:
    """The word runs of an OCR text, case kept.

    A chat message after "name: " is dropped. Handles ("@name", "#tag") and
    separators (``: ; , ! ? | / \\ ( ) [ ] { } < > "``) end a run; so do the
    breaks of :func:`_words` (numbers, gamertags, the user's player tags).
    "[12:03] Ash99 blockedShieldSaver:x1" gives ``[["Ash"], ["blocked",
    "Shield", "Saver"]]`` ("x1" is a number).
    """
    runs: list[list[str]] = [[]]
    for tok in _BREAK.sub(" | ", _CHAT.sub("", text)).split():
        for w in [None] if tok == "|" else _words(tok, tags):
            if w is None:
                runs.append([])
            else:
                runs[-1].append(w)
    return [r for r in runs if r]


def _words(tok: str, tags: tuple[str, ...]) -> list[str | None]:
    """The words of one whitespace token; None marks a break.

    The token is split at other characters (``_ . - '``) and at camelCase
    boundaries ("blockedShieldSaver"). A word with a digit is a break: digits
    only, ordinals ("3RD") and one letter joined to digits (the multiplier
    "x2") whole; a longer letter run and one digit run ("ROUND5", "5KILLS")
    keep their letters. Any other token with a digit is a
    gamertag ("N0VA", "A1B2", or "XX_ASH99", whose separator makes two
    pieces) and a break as a whole, as is a token or camelCase piece that is
    one of the user's player ``tags`` (:func:`~cutpoint.events.killfeed.is_player`).
    """
    if tags and any(similar(tok, t) >= FUZZY for t in tags):
        return [None]  # the token is a player tag
    pieces = re.sub(r"[^A-Za-z0-9]", " ", tok).split()
    if len(pieces) > 1 and any(c.isdigit() for c in tok):
        return [None]  # "XX_ASH99": a gamertag
    out: list[str | None] = []
    for w in (c for p in pieces for c in _CAMEL.split(p)):
        if tags and is_player(w, tags):
            out.append(None)
        elif not any(c.isdigit() for c in w):
            out.append(w)
        elif (
            (m := _NUMBERED.fullmatch(w))
            and not _ORDINAL.fullmatch(w.upper())
            and len(m[1] or m[2]) > 1
        ):
            out += [m[1], None] if m[1] else [None, m[2]]
        else:
            out.append(None)  # digits, "x2", an ordinal or a gamertag ("N0VA")
    return out


__all__ = ["BACKEND", "VERSION", "build_profile", "match_hint", "mine_phrases"]
