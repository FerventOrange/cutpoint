"""``find_words``: where the footage says a word or a phrase, to the word, from the cached
transcript.

``find_moments`` and ``get_transcript`` read moments and phrase-long transcript lines; they
cannot place a single word, nor find a short one such as "a". :func:`find` reads the words
analysis cached for each source (Whisper's per-word times, :meth:`EdgePlanner.words
<cutpoint.assistant.edges.EdgePlanner.words>`; nothing is transcribed) and lists every place the
words of a query are said one after the other: exact words, any case, punctuation left out,
short and common words included. A query in a script written without spaces (Chinese,
Japanese, ...) matches the words' letters run together.

Each hit gives the source, the first word's start and the last word's end (source seconds) and
a few words around it. The model puts a hit into the sequence with an ``add_range`` op
(:mod:`cutpoint.assistant.ops`): an exact row, cut at those times with hard cuts, so words said
in different places can be joined into a sentence ("make her say ...").
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

NAME = "find_words"
JSON_TOOL = f"{NAME} {{query, limit?}}"
DEFAULT_LIMIT, MAX_LIMIT = 10, 30
MAX_QUERY_WORDS = 12
CONTEXT_WORDS = 4  # words shown before and after a hit
# Thai, Lao, Myanmar, Khmer, Hangul Jamo, Kana, CJK, Hangul syllables, CJK compatibility (as
# context.MomentTable.find)
_UNSPACED = re.compile(
    "[\u0e00-\u0eff\u1000-\u109f\u1100-\u11ff\u1780-\u17ff"
    "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)

TOOL = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Where a word or phrase is said, to the word: every place its words are said one "
            "after the other (any case, punctuation ignored, short words such as 'a' too), "
            "with file, start (first word), end (last word) and the words around it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the word or phrase, as said"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}
RULES = f"""

{NAME}: where words are said, even "a". To make a person say a sentence, add each word in order, from anywhere: {{"op": "add_range", "file": "a.mp4", "start": 1.2, "end": 1.5}} (hit times). Search before saying a word is not said."""


@dataclass(frozen=True)
class Found:
    """What :func:`find` gives: ``hits`` (at most the limit, in source and time order), how many
    places match in all (``total``), and how many sources were searched and have words."""

    hits: list[dict[str, Any]]
    total: int
    sources: int
    with_words: int


def norm(text: Any) -> str:
    """``text`` as :func:`find` compares it: lower case, letters and digits only."""
    return "".join(c for c in str(text).casefold() if c.isalnum())


def _spans(said: list[str], tokens: list[str]) -> list[tuple[int, int]]:
    """``(first, last)`` indexes of ``said`` (normalized words; "" for punctuation, skipped)
    where ``tokens`` are said one after the other. Unspaced tokens match the words' letters run
    together instead."""
    where = [i for i, w in enumerate(said) if w]
    out = []
    if any(_UNSPACED.search(t) for t in tokens):
        target = "".join(tokens)
        for k in range(len(where)):
            run = ""
            for j in range(k, len(where)):
                run += said[where[j]]
                if not target.startswith(run):
                    break
                if run == target:
                    out.append((where[k], where[j]))
                    break
        return out
    n = len(tokens)
    for k in range(len(where) - n + 1):
        if all(said[where[k + j]] == tokens[j] for j in range(n)):
            out.append((where[k], where[k + n - 1]))
    return out


def _around(words: list[dict[str, Any]], first: int, last: int) -> str:
    """The hit's words in brackets with up to :data:`CONTEXT_WORDS` words on each side."""

    def text(ws: list[dict[str, Any]]) -> str:
        return " ".join(str(w.get("word", "")).strip() for w in ws).strip()

    before = text(words[max(0, first - CONTEXT_WORDS) : first])
    after = text(words[last + 1 : last + 1 + CONTEXT_WORDS])
    said = f"[{text(words[first : last + 1])}]"
    return " ".join(x for x in (before, said, after) if x)


def find(
    sources: list[str],
    words_of: Callable[[str], list[dict[str, Any]] | None],
    query: str,
    limit: int = DEFAULT_LIMIT,
) -> Found:
    """Every place of ``sources`` (in their order) where the words of ``query`` are said.

    Args:
        sources: The project's sources.
        words_of: A source's cached words (``word``, ``start``, ``end``), [] or None without.
        query: A word or a phrase.
        limit: Hits listed (the total counts all).
    """
    tokens = [t for t in (norm(w) for w in query.split()[:MAX_QUERY_WORDS]) if t]
    hits: list[dict[str, Any]] = []
    total = with_words = 0
    for source in sources:
        try:
            words = words_of(source) or []
        except Exception:  # noqa: BLE001  a source whose words cannot be read has none
            words = []
        words = [w for w in words if isinstance(w, Mapping) and _times(w)]
        if not words:
            continue
        with_words += 1
        if not tokens:
            continue
        for first, last in _spans([norm(w.get("word", "")) for w in words], tokens):
            total += 1
            if len(hits) < limit:
                hits.append(
                    {
                        "source": source,
                        "start": round(float(words[first]["start"]), 2),
                        "end": round(float(words[last]["end"]), 2),
                        "said": _around(words, first, last),
                    }
                )
    return Found(hits, total, len(sources), with_words)


def _times(w: Mapping[str, Any]) -> bool:
    start, end = w.get("start"), w.get("end")
    ok = all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in (start, end))
    return ok and end >= start  # type: ignore[operator]
