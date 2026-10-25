"""Game profiles: a discovered HUD preset enriched with what its text means.

Region discovery (:mod:`cutpoint.events.discover`) finds *where* HUD text is.
:func:`build_profile` adds what can be read from the same discovery OCR
(``discovery/<game>/ocr.jsonl``), with no model, no network and no new
dependency (the ``ocr`` backend):

* ``candidate_phrases``: recurring upper-case multi-word texts of the banner
  regions (and the generic big-text region), kept only when they fuzzy-occur
  in at least :data:`MIN_BURSTS` distinct bursts. A number OCR joined to a word
  is split off and dropped ("WON ROUND5" is "WON ROUND"); see
  :func:`_split_numbers`. Gamertags (other words mixing letters and digits),
  single words, chat lines, static labels (fuzzy-found in most gameplay
  frames), phrases close to one of
  :data:`~cutpoint.events.regions.GENERIC_BANNERS` and phrases naming one of the
  user's player tags are left out; at most :data:`MAX_PHRASES`.
  They are inactive: a recurring header can also show on screens that are no
  event, so ``banner_phrases`` stays as discovery set it. A later model pass
  promotes confirmed candidates into ``banner_phrases``; a user can promote
  one by hand by moving it there (``notes`` says so in the file).
* ``title`` and ``match`` from the game key. ``title`` is the name the VLM
  prompt already derives from the key; ``match`` is left empty unless the key
  is a plausible game name (see :func:`match_hint`).

Regions are kept exactly as discovery proposed them, and ``feed`` colours,
``view`` and ``hype`` are not set: the OCR text alone does not say what they
should be. The result keeps ``"source": "discovered"`` and records its
provenance as ``"profile": {"backend": "ocr", "version": 1}``.

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
from typing import Any

from cutpoint.events.discover import DiscoverParams, boxes_in
from cutpoint.events.killfeed import fuzzy_contains, is_player, norm, similar
from cutpoint.events.regions import GENERIC_BANNERS, bigtext_region

BACKEND = "ocr"
VERSION = 2  # bump when the OCR backend's output changes
MAX_PHRASES = 12
MIN_BURSTS = 2  # a phrase must be seen in this many distinct bursts
STATIC_SHARE = 0.5  # texts in at least this share of gameplay frames are static labels
FUZZY = 0.8  # same threshold as the OCR layer's phrase matching
MAX_WORDS = 5  # longer texts are sentences (hints, chat), not banners
MAX_CANDIDATES = 100  # most frequent candidates compared fuzzily (bounds the cost)
NOTES = (
    "candidate_phrases were mined from the discovery OCR and are inactive. To activate one,"
    " move it into banner_phrases."
)
# a token holding a number: digits only (spaces from "1/3"), or one letter run and
# one digit run ("ROUND5", "5KILLS")
_NUMBERED = re.compile(r"[\d ]+|[A-Z]+\d+|\d+[A-Z]+")
_DIGITS = re.compile(r"\d+")
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
        player_tags: The user's in-game names; phrases with a word matching one
            (fuzzy, as in the OCR layer) are left out.

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

    Occurrences are fuzzy (:func:`~cutpoint.events.killfeed.fuzzy_contains`) for
    the burst count and for the static-label share alike; each distinct text is
    compared with a candidate at most once, and only the :data:`MAX_CANDIDATES`
    most frequent candidates are compared at all.

    Candidates are :func:`_split_numbers` texts, so "WON ROUND5" gives "WON
    ROUND". Bursts count only texts OCR would match once the phrase is promoted
    (plain :func:`~cutpoint.events.killfeed.norm`, digits kept: "WON ROUND5"
    counts, "WON 10 ROUND" does not); the static-label share uses the split
    texts, so labels with changing numbers stay static. Player tags are checked
    against the words before the split ("ASH99").

    Sorted by the number of bursts they occur in (most first), then by text.
    """
    tags = tuple(player_tags)
    params = preset.get("discovery", {}).get("params", {})
    menu_boxes = int(params.get("menu_boxes", DiscoverParams.menu_boxes))
    frames = [f for f in ocr_samples if len(f.get("boxes", [])) <= menu_boxes]
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
    # distinct split banner texts per frame, norm texts per burst; exact frame counts
    by_frame: dict[str, set[str]] = {}
    by_burst: dict[tuple[str, int], set[str]] = {}
    frames_of: dict[str, set[str]] = {}
    words_of: dict[str, set[str]] = {}  # words before the split, for the tag check
    for f in frames:
        fid, bkey = str(f["id"]), (str(f["source"]), int(f["burst"]))
        for text in {b["text"] for rect in rects for b in boxes_in(f, *rect)}:
            n, tag = _split_numbers(text)
            if raw := norm(text):
                by_burst.setdefault(bkey, set()).add(raw)
            if not n:
                continue
            by_frame.setdefault(fid, set()).add(n)
            if not tag and _is_candidate(text, n):
                frames_of.setdefault(n, set()).add(fid)
                words_of.setdefault(n, set()).update(raw.split())
    static = STATIC_SHARE * max(1, len(frames))
    generic = [norm(g) for g in GENERIC_BANNERS]
    cands = sorted(
        (c for c, fs in frames_of.items() if len(fs) < static),  # exact count: cheap pre-filter
        key=lambda c: (-len(frames_of[c]), c),
    )
    frame_sets = [by_frame[k] for k in sorted(by_frame)]
    burst_sets = [by_burst[k] for k in sorted(by_burst)]
    grams = {t: _bigrams(t) for texts in (*burst_sets, *frame_sets) for t in texts}
    kept: list[tuple[str, int]] = []
    tried = 0
    for c in cands:
        if tried >= MAX_CANDIDATES:
            break
        if any(similar(c, k) >= FUZZY for k, _ in kept):
            continue  # an OCR variant of a more frequent phrase
        if any(similar(c, g) >= FUZZY for g in generic):
            continue  # already matched by every JSON preset
        if tags and any(is_player(w, tags) for w in words_of[c]):
            continue  # names the player
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
    kept.sort(key=lambda p: (-p[1], p[0]))
    return [c for c, _ in kept[:MAX_PHRASES]]


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


def _split_numbers(text: str) -> tuple[str, bool]:
    """``norm(text)`` without numbers, and whether it has a gamertag.

    Each whitespace-separated token with a digit is normalised on its own
    (:func:`~cutpoint.events.killfeed.norm`). Digits only ("5", "1/3") and
    ordinals ("3RD") are dropped; one letter run and one digit run ("ROUND5",
    "5KILLS") keep the letters. Any other token with a digit is a gamertag ("N0VA", "A1B2", or
    "XX_ASH99", whose separator makes two words): it is kept as is and flags the
    text, which then is no candidate.
    """
    words, tag = [], False
    for tok in text.split():
        w = norm(tok)
        if any(c.isdigit() for c in w):
            if _ORDINAL.fullmatch(w):
                w = ""
            elif _NUMBERED.fullmatch(w):
                w = _DIGITS.sub(" ", w)
            else:
                tag = True
        words.append(w)
    return " ".join(" ".join(words).split()), tag


def _is_candidate(text: str, words: str) -> bool:
    """Whether an OCR text can be a banner phrase: upper case, mostly letters
    (numbers left out), :func:`_split_numbers` ``words`` of 2 to
    :data:`MAX_WORDS` words, not chat."""
    if any(c in text for c in ":[]<>@#"):
        return False  # chat lines ("[ALL] name: text"), handles
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) < 0.9 * len(letters):
        return False
    chars = [c for c in text if not c.isspace() and not c.isdigit()]
    return 2 <= len(words.split()) <= MAX_WORDS and len(letters) >= 0.8 * len(chars)


__all__ = ["BACKEND", "VERSION", "build_profile", "match_hint", "mine_phrases"]
