"""Repeat damping: what a source says or shows often is filler, so each time counts less (#1071).

A speaker's habit ("oh my god" 36 times in 3 h) or a caption the game shows every round
("ACTION PHASE") says little about any one moment, while a rare "let's go" or a one-off line
does. Per source, every occurrence of a key (a keyword phrase, a normalised on-screen text)
gets the same factor, from how often the key occurs there::

    allowance = max(REPEAT_FREE_MIN, REPEAT_FREE_PER_H * hours)
    factor    = min(1, sqrt(allowance / count))

A key said up to the allowance keeps its full weight; above it, the key's total over the source
grows only with the square root of its count (``count * factor = sqrt(allowance * count)``), so
the filler still adds something, but each occurrence less. The factor depends only on the count
and the source length, never on where an occurrence is or in which order the parts are read.
Generic: no list of phrases or games.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

# A key may occur this often per hour of a source with its full weight ...
REPEAT_FREE_PER_H = 3.0
# ... and this often in any source, however short (a 5-minute clip with three "let's go").
REPEAT_FREE_MIN = 3


def repeat_factor(count: int, span_s: float) -> float:
    """The weight factor (0-1] of each of ``count`` occurrences of one key in a source
    ``span_s`` seconds long (see the module docstring)."""
    allowance = max(float(REPEAT_FREE_MIN), REPEAT_FREE_PER_H * max(0.0, span_s) / 3600.0)
    return 1.0 if count <= allowance else math.sqrt(allowance / count)


def repeat_factors(keys: Iterable[str], span_s: float) -> dict[str, float]:
    """:func:`repeat_factor` per distinct key of ``keys`` (one entry per occurrence)."""
    return {k: repeat_factor(n, span_s) for k, n in Counter(keys).items()}


def text_key(text: str) -> str:
    """The repeat key of an on-screen text: upper case, letters only, every run of digits one
    ``#`` (``Round 2`` and ``ROUND3`` are ``ROUND#``, ``4VS4`` is ``#VS#``) and no spaces or
    punctuation, so the OCR's spacing variants (``Attackersafeguardsactive``, ``Attacker
    safeguards active``) are one key."""
    return re.sub(r"[^A-Z#]", "", re.sub(r"\d+", "#", text.upper()))
