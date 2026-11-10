"""Spread the kinds of a proposal when the user asks for a mix, in code.

A model piles up the clips of one kind: sixteen riding clips in a row, the funny ones at the end.
When the message asks to break up, mix in or intersperse kinds (:func:`asks_for_mix`), :func:`pace`
reorders the final proposal with ``move`` ops marked ``"auto": true`` so the funny, dialogue and
scenic clips ("breakers") sit between the riding runs, and adds dialogue moments when the message
names dialogue and the sequence has none. Without such wording nothing here runs: pacing is left
to the model.

Every try is measured with ``evaluate`` (the session's own check of ops: the same apply path as
the model's proposal), so nothing here knows how a sequence is applied.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cutpoint.assistant.context import _GENERIC
from cutpoint.assistant.ops import _in_order
from cutpoint.assistant.topup import Evaluate, _overlaps, _round_robin

ACTION, SCENIC, FUNNY, DIALOGUE, CRASH = "action", "scenic", "funny", "dialogue", "crash"
RUNS = (ACTION, CRASH)  # the kinds the breakers go between: a crash is footage of the activity
BREAKERS = (FUNNY, DIALOGUE, SCENIC)  # the order the summary names them in
MAX_DIALOGUE = 3  # dialogue moments added to a sequence that has none
SPOKEN_WORDS = 4  # words over a moment that make it dialogue
CUE_STRONG = 5  # a cue this strong (0-10) names the kind of a moment with a generic label
TRIES = 5  # rows tried as the one a dialogue moment replaces

_NOT_AUDIO = (
    r"(?!\s+(?:(?:some|the|a|my|any|more|less|background|extra) )*"
    r"(?:music|audio|sound|songs?|tracks?|volume|voice|voiceover|sfx|levels?))"
)
# The sport words of "between the <X> runs": the same sports as _ACTION_WORDS.
_RIDE_WORDS = r"ski\w*|snowboard\w*|bik(?:e|es|ing)|rid(?:e|es|er|ers|ing)|mtb|surf(?:s|ing|ers?)?|skat(?:e|es|ing)|trails?"
_MIX = re.compile(
    r"(?<![\w-])(?:"
    r"break(?:s|ing)? (?:it )?up(?!(?:\W+\w+){0,4}?\W+into\b)|break(?:s|ing)? the (?:monotony|runs?)"
    rf"|mix(?:ed|es|ing)? (?:in|of|with){_NOT_AUDIO}|mixture of"
    r"|inters(?:perse|persed|perses|persing)|sprinkl(?:e|es|ed|ing)|scatter(?:s|ed|ing)?"
    r"|occasional(?:ly)?(?! (?:stutter|glitch|flicker|jitter|artifact|noise|blur|drop))"
    r"|now and (?:then|again)|from time to time"
    rf"|in[- ]between (?:the |some |all |each )?(?:{_RIDE_WORDS}|runs?)"
    rf"|between (?:the |each |every |some )?(?:{_RIDE_WORDS}(?: (?:runs?|shots?|clips?))?|runs?)"
    r"|alternat(?:e|es|ed)(?! (?:take|version|cut|clip|shot|angle|ending|option))|alternating"
    r"|interleav(?:e|es|ed|ing)|vary(?:ing)? (?:up )?the pac(?:e|ing)"
    r"|change of pace|pepper(?:s|ed|ing)?|punctuat(?:e|es|ed|ing)"
    r")(?![\w-])",
    re.I,
)
_DIALOGUE = re.compile(
    r"(?<![\w-])(?:dialogu?e|conversations?|banter|commentary"
    r"|(?:talk|talking|chat|chatting) (?:clips?|moments?|scenes?|parts?|bits?|segments?|shots?)"
    r"|(?:some|more|the|any) (?:talk|talking|chat)(?! about))"
    r"(?![\w-])",
    re.I,
)
# A request that is said not to be wanted: "don't break up", "no dialogue", "without alternating".
_NEG = re.compile(
    r"(?<![\w-])(?:don'?t|do\s+not|never|no|not|without|stop|avoid)(?![\w-])(?:\W+\w+){0,3}\W*$",
    re.I,
)
# A transition is no mix: "a fade in between the clips", "a crossfade between the runs".
_TRANSITION = re.compile(
    r"(?<![\w-])(?:fade|crossfade|transition|dissolve|cut)s?(?![\w-])(?:\W+\w+){0,3}\W*$", re.I
)


_CLAUSE = re.compile(r"[,;:.!?]|(?<![\w-])(?:but|just|instead|rather|then)(?![\w-])", re.I)


def _wanted(pattern: re.Pattern[str], text: str, *, transitions: bool = False) -> bool:
    """Whether ``pattern`` matches ``text`` somewhere that the words before it do not negate
    (or, with ``transitions``, turn into a fade or cut)."""
    for m in pattern.finditer(text):
        # only the clause the match is in: a "no" or "fade" before a comma or "but" is not about it
        before = _CLAUSE.split(text[max(0, m.start() - 40) : m.start()])[-1]
        if not _NEG.search(before) and not (transitions and _TRANSITION.search(before)):
            return True
    return False


def asks_for_mix(text: str) -> bool:
    """Whether the message asks to break up, mix in, intersperse or alternate kinds ("an
    occasional funny moment to break up the riding"). An audio mix ("mix the audio", "no music
    mix"), a transition between clips, and a request that is negated ("don't break up the
    riding") are none."""
    return _wanted(_MIX, text, transitions=True)


def names_dialogue(text: str) -> bool:
    """Whether the message asks for dialogue (dialogue, conversation, banter, commentary, talk or
    chat moments), unless the words before it negate it ("no dialogue")."""
    return _wanted(_DIALOGUE, text)


_SCENIC_WORDS = frozenset(
    [
        "view",
        "views",
        "scenic",
        "scenery",
        "beauty",
        "beautiful",
        "vista",
        "vistas",
        "landscape",
        "panorama",
        "panoramic",
    ]
)
_FUNNY_WORDS = frozenset(
    [
        "laugh",
        "laughs",
        "laughing",
        "laughter",
        "funny",
        "joke",
        "jokes",
        "hilarious",
        "giggle",
        "giggles",
        "comedy",
    ]
)
_ACTION_WORDS = frozenset(
    [
        "ski",
        "skis",
        "skiing",
        "skier",
        "skiers",
        "fast",
        "speed",
        "speeding",
        "crash",
        "crashes",
        "wipeout",
        "cruise",
        "cruising",
        "crew",
        "jump",
        "jumps",
        "airtime",
        "air",
        "carve",
        "carving",
        "run",
        "runs",
        "descent",
        "powder",
        "ride",
        "riding",
        "groomer",
        "trees",
        "bike",
        "biking",
        "rider",
        "riders",
        "mtb",
        "trail",
        "trails",
        "surf",
        "surfing",
        "skate",
        "skating",
        "snowboard",
        "snowboarding",
    ]
)
_TALK_WORDS = frozenset(
    [
        "talk",
        "talking",
        "speech",
        "dialogue",
        "dialog",
        "conversation",
        "banter",
        "chat",
        "commentary",
        "voice",
        "speaking",
    ]
)
_CUE_KIND = {
    **dict.fromkeys(("beauty", "scenic", "scenic_vista"), SCENIC),
    **dict.fromkeys(("funny", "laughter", "voice_laughter"), FUNNY),
    "standing_talking": DIALOGUE,
    **dict.fromkeys(
        (
            *("airtime", "crash", "wipeout", "speed", "carve"),
            *("jump_air", "landing", "crash_fall", "spray", "fast_open", "trees"),
        ),
        ACTION,
    ),
}


def moment_kind(
    label: str,
    cand: Mapping[str, Any] | None,
    cues: Mapping[str, int],
    spoken: Callable[[], bool],
) -> str:
    """The kind of a moment: ``crash``, ``scenic``, ``funny``, ``dialogue`` or ``action``.

    A crash moment comes first: ``kind`` crash or wipeout (a crash moment, its reaction, an
    action-cam crash), or a forced ``crash`` record (the crash behind it, at or above the
    analysis' must-include confidence). A moment a weaker crash only marked keeps its kind. The words of its labels and ``kind`` decide first (visual pool ``view`` is scenic, ``crew`` and
    ``cruise`` are action; ``beauty``/``view``/``scenic``; ``laugh``/``funny``; ``fast``, ``speed``,
    ``crash``, ``jump``, ...; ``talk``/``voice``). Then the strongest cue of a generic label
    (``action-cam-moment``, ``loud-moment``) decides, then speech over it (``spoken``: dialogue), else it
    is action. A moment a talk stretch nominated (``signals.events.nominated`` holds
    ``dialogue-*``) is dialogue; one a stretch only widened keeps its own type, from the label it
    had before it took the stretch's title.
    """
    cand = cand or {}
    if cand.get("kind") in (CRASH, "wipeout") or (
        cand.get("crash") and cand.get("crash_forced", bool(cand.get("must_include")))
    ):
        return CRASH
    events = (cand.get("signals") or {}).get("events") or {}
    if any(str(n).startswith("dialogue-") for n in events.get("nominated") or []):
        return DIALOGUE  # made from a talk stretch, whatever its title says
    kind = cand.get("kind") if isinstance(cand.get("kind"), str) else ""
    named = (
        label,
        cand.get("label"),
        cand.get("vision_label"),
    )  # "action-cam-moment" names nothing
    if cand.get(
        "dialogue_title"
    ):  # a moment a stretch widened keeps its own type: not the title's words
        was = cand.get("widened_from") or {}
        named = (was.get("label") or cand.get("label"), cand.get("vision_label"))
    text = " ".join([*(str(x) for x in named if x and not _GENERIC.fullmatch(str(x))), kind])
    tokens = set(re.split(r"[^a-z]+", text.lower()))
    if kind == "view" or tokens & _SCENIC_WORDS:
        return SCENIC
    if tokens & _FUNNY_WORDS:
        return FUNNY
    if kind in ("crew", "cruise") or tokens & _ACTION_WORDS:
        return ACTION
    if tokens & _TALK_WORDS:
        return DIALOGUE
    for cue, value in cues.items():  # strongest first
        if value >= CUE_STRONG and cue in _CUE_KIND:
            return _CUE_KIND[cue]
    return DIALOGUE if spoken() else ACTION


def _even(n: int, parts: int) -> list[int]:
    """``n`` split into ``parts`` sizes that differ by at most one, the larger first."""
    return [n // parts + (j < n % parts) for j in range(parts)]


def _interleaved(queues: Mapping[str, list[int]]) -> list[int]:
    """The items of ``queues`` mixed so that each kind is spread evenly (the kind with the
    largest share left goes next), each kind in its own order."""
    left = {k: list(v) for k, v in queues.items()}
    total = {k: len(v) for k, v in queues.items()}
    out: list[int] = []
    while any(left.values()):
        pick = max((k for k in left if left[k]), key=lambda k: len(left[k]) / total[k])
        out.append(left[pick].pop(0))
    return out


def arrange(kinds: Sequence[str]) -> list[int]:
    """The order, as indexes into ``kinds``, that puts the breakers (scenic, funny, dialogue)
    between the action runs: a scenic first row stays first, each kind keeps its order, the
    kinds are spread evenly and the action clips fall into as many runs as there are breakers
    (an even split, so a run is short whenever the breakers are enough). The order is
    unchanged without action clips or without breakers."""
    keep = [0] if kinds and kinds[0] == SCENIC else []
    action = [i for i, k in enumerate(kinds) if k in RUNS]
    queues: dict[str, list[int]] = {}
    for i, k in enumerate(kinds):
        if k not in RUNS and i not in keep:
            queues.setdefault(k, []).append(i)
    breakers = _interleaved(queues)
    if not action or not breakers:
        return list(range(len(kinds)))
    runs = min(len(breakers) + 1, len(action))
    sizes = _even(len(action), runs)
    gaps = _even(len(breakers), runs - 1) if runs > 1 else [0]
    out, a, b = list(keep), 0, 0
    for j, size in enumerate(sizes):
        out += action[a : a + size]
        a += size
        if j < len(gaps):
            out += breakers[b : b + gaps[j]]
            b += gaps[j]
    return out + breakers[b:]  # only with a single action clip


def longest_run(kinds: Sequence[str]) -> int:
    """The most action clips in a row."""
    best = run = 0
    for k in kinds:
        run = run + 1 if k in RUNS else 0
        best = max(best, run)
    return best


def move_ops(current: Sequence[str], target: Sequence[str]) -> list[dict[str, Any]]:
    """``move`` ops (marked auto) that turn the order ``current`` into ``target`` (the same
    ids), moving as few rows as possible: the longest run already in order stays."""
    stay = _in_order(list(target), {sid: (i, None) for i, sid in enumerate(current)})
    rows = list(current)
    out: list[dict[str, Any]] = []
    for i, sid in enumerate(target):
        if sid in stay:
            continue
        rows.remove(sid)
        to = 1 if i == 0 else rows.index(target[i - 1]) + 2  # right after the row before it
        rows.insert(to - 1, sid)
        out.append({"op": "move", "id": sid, "to": to, "auto": True})
    return out


@dataclass
class Pace:
    """What :func:`pace` did: the ``checked`` proposal with the extra ops, the ids ``moved``,
    ``added`` (dialogue) and ``replaced`` (action clips dropped for them), and how many
    breakers of each kind are spread (``spread``)."""

    checked: dict[str, Any]
    moved: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    spread: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.moved) + len(self.added) + len(self.replaced)


def _score(item: Mapping[str, Any] | None) -> float:
    v = (item or {}).get("score")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def pace(
    checked: dict[str, Any],
    *,
    kind_of: Callable[[Mapping[str, Any]], str],
    evaluate: Evaluate,
    candidates: Mapping[str, Mapping[str, Any]],
    searched: Sequence[Sequence[str]],
    dialogue: bool,
    upper: float | None,
    removable: Collection[str] = (),
) -> Pace | None:
    """Spread the breakers of ``checked`` (a proposal that ``evaluate`` made: ``ops``, ``after``,
    ``length``) between its action runs; None when nothing changes.

    Args:
        checked: The proposal to pace.
        kind_of: The kind of a moment or sequence row (:func:`moment_kind`).
        evaluate: ``ops -> checked`` for the whole op list on the turn's sequence, or None when
            they do not apply.
        candidates: The moments ``add`` can take, by id.
        searched: The ids each of this turn's searches found, one list per search.
        dialogue: The message names dialogue: with none in the sequence, add up to
            :data:`MAX_DIALOGUE` dialogue moments (the searches' first, else the best-scoring
            unused ones) while the length stays within ``upper``; when one would not fit, it
            replaces the lowest-scoring action clip of a similar length among ``removable``,
            and with none of those it is not added.
        upper: The most seconds the sequence may last (None: no limit).
        removable: Ids of the clips code may remove (the ones the auto-fill added this turn);
            the user's and the model's picks are never removed.
    """
    base = list(checked["ops"])
    extra: list[dict[str, Any]] = []
    cur = checked
    result = Pace(checked)

    def attempt(new: list[dict[str, Any]], *, limit: bool = False) -> bool:
        nonlocal cur
        got = evaluate([*base, *extra, *new])
        if got is None or (limit and upper is not None and got["length"] > upper):
            return False
        extra.extend(new)
        cur = got
        return True

    if dialogue and not any(kind_of(r) == DIALOGUE for r in cur["after"] or []):
        hits = list(_round_robin(searched))
        rest = sorted(candidates, key=lambda c: -_score(candidates[c]))
        wanted = (c for c in dict.fromkeys([*hits, *rest]) if c in candidates)
        for cid in wanted:
            if len(result.added) >= MAX_DIALOGUE:
                break
            cand = candidates[cid]
            rows = cur["after"] or []
            if kind_of(cand) != DIALOGUE or any(r["id"] == cid for r in rows):
                continue
            if _overlaps(cand, rows):
                continue
            if attempt([{"op": "add", "id": cid, "auto": True}], limit=True):
                result.added.append(cid)
                continue
            if upper is None:
                continue
            size = float(cand["end"]) - float(cand["start"])
            action = [
                (i, r)
                for i, r in enumerate(rows, 1)
                if kind_of(r) in RUNS
                and r["id"] in removable
                and abs((r["end"] - r["start"]) - size) <= max(3.0, size / 2)
            ]
            action.sort(key=lambda x: _score(candidates.get(x[1]["id"])))
            for i, row in action[:TRIES]:
                swap = [
                    {"op": "remove", "id": row["id"], "auto": True},
                    {"op": "add", "id": cid, "at": i, "auto": True},
                ]
                if attempt(swap, limit=True):
                    result.added.append(cid)
                    result.replaced.append(row["id"])
                    break

    rows = cur["after"] or []
    kinds = [kind_of(r) for r in rows]
    order = arrange(kinds)
    ids = [r["id"] for r in rows]
    target = [ids[i] for i in order]
    if longest_run([kinds[i] for i in order]) < longest_run(kinds):  # else it is spread enough
        moves = move_ops(ids, target)
        if attempt(moves):
            result.moved = [op["id"] for op in moves]
        elif attempt([{"op": "order", "ids": target, "auto": True}]):  # too many ops: one order
            stay = _in_order(target, {sid: (i, None) for i, sid in enumerate(ids)})
            result.moved = [sid for sid in target if sid not in stay]  # not rows that only shifted
        if result.moved:
            keep = kinds[0] == SCENIC
            for k in BREAKERS:
                n = sum(x == k for x in kinds) - (keep and k == SCENIC)
                if n:
                    result.spread[k] = n
    if not result.count:
        return None
    result.checked = cur
    return result
