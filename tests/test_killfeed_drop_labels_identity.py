"""``Interpreter.drop_labels`` compares each distinct pair of names once (#1085, A703): it must
keep and drop the same icon kills as the old every-line-with-every-line loop, frozen here
(synthetic names, invented; no OCR)."""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from cutpoint.events.killfeed import (
    LABEL_REPEATS,
    ROW_NOTICE_MAX_S,
    STATIC_FEED_GUARD_S,
    Interpreter,
    OcrParams,
    _runs,
    _two_hues,
)

# ------------------------------------------------------------------ the old code, frozen


def _old_drop_labels(interp: Interpreter) -> None:
    tracker = interp.tracker
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


# ------------------------------------------------------------------ the corpus

PLAYERS = [
    "Alderwick", "Boxer7", "Kip", "Nova", "DramaticRail39", "Marigold",
    "Quillmere77", "Сокол", "ドラゴン", "Ünal_K", "Straße", "हिन्दी",
]  # fmt: skip
# two words that come back together; some are near a player's name ("Novak", "Kipper")
LABELS = [("Defuse", "Planted"), ("Round", "Won"), ("Novak", "Kipper"), ("СОКОЛ", "Nova")]
HUES = ["blue", "orange", "white", None]


def _variant(rng: random.Random, name: str) -> str:
    """The name as OCR may read it: misread letters, "rn" for "m", a letter dropped, a cut
    read (nested in the whole name), two letter swaps (only similar), a clan tag, case or
    punctuation only."""
    r = rng.random()
    if r < 0.34:
        return name
    if r < 0.4 and len(name) > 6:
        c = list(name)
        c[1], c[2], c[-3], c[-2] = c[2], c[1], c[-2], c[-3]
        return "".join(c)
    if r < 0.5:
        return name.translate(str.maketrans("oOiIlS", "001115"))
    if r < 0.56:
        return name.replace("m", "rn")
    if r < 0.66 and len(name) > 3:
        i = rng.randrange(len(name))
        return name[:i] + name[i + 1 :]
    if r < 0.78 and len(name) > 3:
        return name[: rng.randint(max(2, len(name) * 6 // 10), len(name) - 1)]
    if r < 0.86:
        return f"[{rng.choice(['ABC', 'XO'])}] {name}"
    if r < 0.93:
        return name.lower()
    return f"{name}{rng.choice(['.', '_', '™'])}"


def _corpus(n: int, seed: int) -> Interpreter:
    rng = random.Random(seed)
    interp = Interpreter(OcrParams(), 1.0)
    t = 0.0
    while len(interp.tracker.events) < n:
        t += rng.uniform(9.0, 14.0)
        if rng.random() < 0.3:
            killer, victim = rng.choice(LABELS)
            if rng.random() < 0.3:
                killer = _variant(rng, killer)
            hk = hv = rng.choice(["white", None])
        else:
            killer, victim = rng.sample(PLAYERS, 2)
            killer, victim = _variant(rng, killer), _variant(rng, victim)
            hk, hv = rng.choice(HUES), rng.choice(HUES)
        if rng.random() < 0.03:
            victim = "~~"  # no letters: matches nothing, not even itself
        sightings = 26 if rng.random() < 0.04 else rng.randint(1, 4)
        for k in range(sightings):
            interp.tracker.see(
                t + k, "kill", f"{killer} KILL".upper(), exclusive=True,
                killer=killer, victim=victim, entry="icon", text=f"{killer} > {victim}",
                killer_hue=hk, victim_hue=hv, confidence=round(rng.uniform(0.6, 1), 3),
                region="feed",
            )  # fmt: skip
        t += sightings
    # at the repeat count and one below it, misread once; a third read only similar
    for killer, victim in [
        ("Zorb", "Hexa"), ("Z0rb", "Hexa"), ("Zorb", "Hexa"),
        ("Pirate", "Quill"), ("Pirate", "Qu1ll"),
        ("Ravenholt88", "Bracketon"), ("Ravenholt88", "Bracketon"), ("Rvaenhol8t8", "Bracketon"),
    ]:  # fmt: skip
        t += 12.0
        interp.tracker.see(
            t, "kill", f"{killer} KILL".upper(), exclusive=True, killer=killer, victim=victim,
            entry="icon", text=f"{killer} > {victim}", region="feed",
        )  # fmt: skip
    # events that are no icon kills stay whatever the names
    interp.tracker.events.insert(3, {"type": "kill", "entry": "text", "killer": "Kip",
                                     "victim": "Nova", "_times": [1.0], "t": 1.0})  # fmt: skip
    interp.tracker.events.append({"type": "banner", "text": "Defuse", "_times": [t], "t": t})
    return interp


@pytest.mark.parametrize("seed", range(6))
def test_drop_labels_keeps_and_drops_what_the_old_loop_did(seed: int) -> None:
    new = _corpus(120, seed)
    old = copy.deepcopy(new)
    before = len(new.tracker.events)
    new.drop_labels()
    _old_drop_labels(old)
    assert new.tracker.events == old.tracker.events  # same events, order and fields
    assert 0 < len(new.tracker.events) < before  # the corpus drops some and keeps some
