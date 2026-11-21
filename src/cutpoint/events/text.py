"""HUD text normalised for comparisons, in any script (#1085, A137)."""

from __future__ import annotations

import re
import unicodedata

_SPACES_RE = re.compile(r"\s+")
_NOT_ASCII_ALNUM_RE = re.compile(r"[^A-Z0-9 ]")


def alnum_words(text: str) -> str:
    """The letters and digits of ``text`` in any script, upper-case, with every other run of
    characters as one space and none at either end ("Kip_V2 (x)" -> "KIP V2 X").

    ASCII text gives upper-case A-Z and 0-9 as it always did. In other text the letters and
    digits are NFKC-normalised, so the fullwidth, superscript and circled digits OCR can read are
    plain ("²" "2", "①" "1"), then case-folded and upper-cased, so "é" and "É" are one and "ß"
    is "SS". Combining marks (Unicode category M, the vowel signs of "हिन्दी") stay. Symbols
    go before NFKC, so "Kip™" is "KIP", not "KIPTM".
    """
    if text.isascii():
        return _SPACES_RE.sub(" ", _NOT_ASCII_ALNUM_RE.sub(" ", text.upper())).strip()
    return _alnum_words_any(text)


def _alnum_words_any(text: str) -> str:
    """:func:`alnum_words` for any text (the ASCII rule is a fast path of it)."""
    kept = _spaced(unicodedata.normalize("NFC", text))
    return " ".join(_spaced(unicodedata.normalize("NFKC", kept).casefold().upper()).split())


def _spaced(text: str) -> str:
    """``text`` with each character that is no letter, digit or combining mark as a space; a mark
    after a space or at the start has nothing to attach to and is a space too."""
    out: list[str] = []
    for c in text:
        if c.isalnum() or (unicodedata.category(c).startswith("M") and out and out[-1] != " "):
            out.append(c)
        else:
            out.append(" ")
    return "".join(out)


def notice_letters(text: str) -> bool:
    """The OCR layer's rule for an unknown notice line, shared by the notice regions, the feed
    rows' notices and discovery (#1085 A339): 4 or more letters, 60% of the characters (spaces
    left out)."""
    n = sum(ch.isalpha() for ch in text)
    return n >= 4 and n >= 0.6 * len(text.replace(" ", ""))
