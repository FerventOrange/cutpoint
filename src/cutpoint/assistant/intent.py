"""What a chat message or a model's reply says, read by English-only heuristics: the length a
message asks for (:func:`target_seconds`) or a proposal's summary claims (:func:`claimed_seconds`,
:func:`named_seconds`), whether a message asks for an edit, advice or a removal
(:func:`edits`, :func:`asks_advice`, :func:`wants_removal`), a mix of kinds or dialogue
(:func:`asks_for_mix`, :func:`names_dialogue`), and whether a reply promises work it does not do
(:func:`promises`, :data:`ANNOUNCES`). Regular expressions over English wording, so other languages
read as no intent; issue #1167 proposes structured output from the model instead.
"""

from __future__ import annotations

import re

_COUNTS = {
    "a": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}
_NUM = r"(\d+(?:[.,]\d+)?|(?:" + "|".join(_COUNTS) + r")\b)"
_UNIT = r"\s*-?\s*(minutes?|mins?|seconds?|secs?|s)\b"
# A length is a duration the message gives to the whole sequence: after a word that says so ("make
# it ~4 min", "get it to 4 minutes", "total of 240 s", "under 2 minutes"), or before one ("4 minutes
# long", "a 90 second reel"). A place or a clip in the video is none: "the clip at 3 minutes",
# "the 2 minute clip", "the first 5 minutes", "around 4:20".
_VIDEO = (
    r"(?:video|montage|reel|edit|cut|sequence|highlights?|compilation|film|movie|trailer|recap)"
)
# "make the video", "the video should be", "keep it to": the length of the whole sequence
_SEQ_CUE = (
    rf"make (?:the|this|that) {_VIDEO}(?: be)?|(?:the |this )?{_VIDEO} (?:should|must|needs to|has to)"
    r" be|it should be|keep it (?:to|at)"
)
_LEN_CUE = (
    r"(?:make it|get it to|bring it to|cut it to|cut to|trim it to|keep it to|keep it under|build"
    r"|aim(?:ing)? for|overall|target(?:ing)?(?: of)?|total(?:ling)? of|totall?ing|want|need|~|about|around"
    r"|approx\w*|roughly|nearly|almost|under|at most|up to|at least|no more than|no longer than"
    rf"|{_SEQ_CUE})"
)
_LEN_NOUN = (
    r"(?:video|montage|reel|edit|cut|highlights?|compilation|sequence|film|movie|trailer|vlog"
    r"|recap|summary|version)"
)
_BEFORE = re.compile(rf"(?<![\w:.]){_LEN_CUE}\s*(?:be\s+|a\s+|an\s+)?~?\s*{_NUM}{_UNIT}", re.I)
_AFTER = re.compile(
    rf"(?<![\w:.]){_NUM}{_UNIT}(?:\s+(?:long|total|in total|overall)\b|\s+{_LEN_NOUN}\b)", re.I
)
_CLOCK_CUE = (
    r"(?:make it|get it to|bring it to|total of|target|aim for|keep it under|under|at most|up to"
    rf"|no more than|want|need|{_SEQ_CUE})"
)
_CLOCK = re.compile(
    rf"(?<![\w:.])(?:{_CLOCK_CUE}\s*(?:be\s+)?(\d{{1,2}}):([0-5]\d)"
    r"|(\d{1,2}):([0-5]\d)\s+(?:long|total|in total))(?![\w:])",
    re.I,
)
# A change to a clip ("by 2 minutes", "trim 5 s") is no length, nor is one named by its id
# ("extend m3 to 60 s").
_CHANGE = re.compile(r"\b(?:by|extra|another|add|more|trim|cut|remove|plus|extend|each)\W*$", re.I)
_SEQ_NOUN = re.compile(
    r"\b(?:whole|total|overall|video|montage|reel|sequence|highlights?|compilation|film|movie"
    r"|trailer|recap)\b",
    re.I,
)
_CLIP_REF = re.compile(rf"\bm\d+\b(?:(?!{_SEQ_NOUN.pattern})[^.!?;\n])*$", re.I)
# How far back from a length the checks look: they run per match, so the whole text would make a
# message with many lengths quadratic.
_LOOKBACK = 300
# A length for each clip ("each clip about 5 s", "clips around 8 seconds", "10 s clips") or a change
# ("30 s more", "5 s longer") is none either.
_SEQ_WORD = rf"\b(?:it|whole|total|overall|{_VIDEO})\b"
_SEQ = re.compile(_SEQ_WORD, re.I)
_PER_CLIP_BEFORE = re.compile(
    rf"\b(?:each|every|per)\b(?:\s+(?:one\s+)?(?:of\s+(?:the|these|those)\s+)?{_VIDEO}\b)?"
    rf"(?:(?!{_SEQ_WORD}|{_NUM}{_UNIT})[^.!?;\n])*$",
    re.I,
)
# The text before a length cue ("make the clips about 5 s long": the cue "about" ends it)
_CUE_END = re.compile(rf"{_LEN_CUE}\s*(?:be\s+|a\s+|an\s+)?~?\s*$", re.I)
# Right after "clips": seconds are per clip, minutes the whole video ("clips around 2 minutes").
_CLIPS_BEFORE = re.compile(
    r"\bclips?\s*(?:(?:should|must|needs? to|have to)\s+be\s+|are\s+|of\s+|be\s+)?$", re.I
)
_PER_CLIP_AFTER = re.compile(
    r"\s*(?:each|per clip|a clip|clips?|more(?!\s+or\s+less)|extra|longer|shorter)\b", re.I
)
# "4 min and 30 s": the seconds that follow minutes
_AND_SECONDS = re.compile(rf"\s*(?:and\s+)?{_NUM}{_UNIT}", re.I)


def _count(word: str) -> float:
    word = word.lower()
    return float(_COUNTS[word]) if word in _COUNTS else float(word.replace(",", "."))


def target_seconds(text: str) -> float | None:
    """The length in seconds that the user's message gives the sequence ("make it ~4 min", "get
    it to 4 minutes", "a 90 second reel", "total of 240 s", "under 2 minutes", "make it 1:30"),
    else None. A place or a clip in the video ("remove the clip at 3 minutes", "the first 5
    minutes", "around 4:20") and a change to a clip ("by 2 minutes") are no length."""
    for pattern in (_BEFORE, _AFTER):
        for m in pattern.finditer(text):
            before = text[max(0, m.start() - _LOOKBACK) : m.start()]
            if _CHANGE.search(before) or (
                _CLIP_REF.search(before) and not _SEQ_NOUN.search(text[m.start() : m.end()])
            ):
                continue
            n, unit = _count(m.group(1)), m.group(2).lower()
            stems = (before, _CUE_END.sub("", before))
            if _PER_CLIP_BEFORE.search(
                text[max(0, m.start(1) - _LOOKBACK) : m.start(1)]
            ) and not _SEQ.search(text[m.end(2) : m.end()]):
                continue
            if _PER_CLIP_AFTER.match(text, m.end()):
                continue
            if not unit.startswith("m") and any(_CLIPS_BEFORE.search(s) for s in stems):
                continue
            if not unit.startswith("m"):
                return n
            more = _AND_SECONDS.match(text, m.end())
            if more and more.group(2).lower().startswith("s"):
                return n * 60 + _count(more.group(1))
            return n * 60
    for m in _CLOCK.finditer(text):
        before = text[max(0, m.start() - _LOOKBACK) : m.start()]
        if not _CLIP_REF.search(before) or _SEQ_NOUN.search(text[m.start() : m.end()]):
            groups = [g for g in m.groups() if g is not None]
            return int(groups[0]) * 60 + int(groups[1])
    return None


# A length a proposal's summary gives the whole sequence: tied to a cue ("now about 34 s", "a total
# of 90 s", "the video is 1:30", "brings it to 2 min", "now runs 1:30", "90 s in all"). "under 1
# minute", a length without a cue ("added about 12 s") and a place ("now at 0:12") are no claim.
_CLAIM_NUM = _NUM.replace("(", "(?P<n>", 1)
_CLAIM_UNIT = _UNIT.replace("(", "(?P<u>", 1)
_CLAIM_LEN = rf"(?:{_CLAIM_NUM}{_CLAIM_UNIT}|(?P<mm>\d{{1,2}}):(?P<ss>[0-5]\d)(?![\d:]))"
_CLAIM_FILL = (
    r"(?:\s*(?:(?:about|around|roughly|approx\w*|nearly|almost|just|at|to|of|is|now|runs|lasts"
    r"|comes\s+to|for|a)\b|~))*\s*"
)
_CLAIM_BEFORE = re.compile(
    rf"(?<![\w])(?P<cue>now(?!\s+at\b)|total(?:l?ing)?|overall|in all|brings? it to|bringing it to|runs|lasts"
    rf"|{_VIDEO}\s+(?:is|runs|lasts))\b{_CLAIM_FILL}{_CLAIM_LEN}",
    re.I,
)
_CLAIM_AFTER = re.compile(rf"(?<![\w:.]){_CLAIM_LEN}\s+(?:in all|in total|total|overall)\b", re.I)
# A cue that needs the sequence as its subject: "the jump is now 8 s", "m3 now runs 12 s", "each
# clip is now 5 s" and "Extended the jump; it is now 8 s" give a clip's length.
_SUBJECT_CUES = ("now", "runs", "lasts")
_CLAUSE = re.compile(r"[^.;!?\n()]*$")
_SENTENCE = re.compile(r"[^.!?\n]*$")
_OTHER_SUBJECT = re.compile(
    rf"\bm\d+\b|\b(?:each|every|per)\b|\b(?:the|this|that)\s+(?!{_VIDEO}\b)\w+\s+(?:is|was|runs"
    r"|lasts)\b",
    re.I,
)
_CLIP_CHANGE = re.compile(r"\b(?:extend|lengthen|trimm?|shorten)\w*", re.I)
# A subject cue with the sequence named as its subject ("the montage now lasts 34 s"), or another
# ("the laugh now runs 12 s", "the jump now lasts 8 s")
_VIDEO_SUBJECT = re.compile(rf"\b(?:the|this|that|whole|final)\s+{_VIDEO}\b", re.I)
_NOUN_NOW = re.compile(
    rf"\b(?:the|this|that)\s+(?!{_VIDEO}\b|whole\b|final\b)\w+\s+(?:now\b|(?:is|was|runs|lasts)\b)",
    re.I,
)
# After a length: a change ("12 s longer"), a clip's ("5 s each") or a place in the source ("12:30
# in the source") is no sequence length.
_NOT_SEQ_AFTER = re.compile(
    r"\s*(?:long\s+)?(?:longer|shorter|more|less|extra|each|apiece|per\b|a clip|a piece"
    r"|(?:in|of|from|into)\s+the\s+(?:source|recording|stream|original|footage|vod))\b",
    re.I,
)
# Before a length in its clause: a change that takes it as object ("Cut 20 s in total", "Removed a
# total of 20 s", "Added three clips totalling 15 s")
_CHANGE_VERB = re.compile(
    r"\b(?:add(?:ed|ing|s)?|cut(?:ting|s)?|trimm?(?:ed|ing|s)?|remov\w*|delet\w*|dropp?\w*"
    r"|extend\w*|lengthen\w*|shorten\w*|shav\w*|sav(?:e|ed|es|ing)|took|tak(?:e|es|ing)"
    r"|gain\w*|lost|los(?:e|es|ing))\b",
    re.I,
)
# Between the change and the length, a word that makes the length the sequence's ("Added the laugh
# for a total of 90 s", "Trimmed m3, bringing it to 34 s")
_CHANGE_TO = re.compile(rf"\b(?:for|to|into|bringing|making|so|now|it|{_VIDEO})\b", re.I)
_PART = re.compile(r"[^.;:,!?\n()]*$")
# After a length, up to the next number or the end of its part
_PART_AFTER = re.compile(r"[^.;:,!?\n()\d]*")
# The next part, when it is about the length ("60 s, which you asked for", "60 s total, you would
# need more footage", "60 s, as requested")
_NEXT_PART = re.compile(r",\s*((?:which|that|as|you\w*|requested|asked)\b[^.;:,!?\n()]*)", re.I)
# In a length's part: a length asked for, negated or hypothetical, not the sequence's current one
# ("You asked for a total of 60 s", "not the 60 s overall you wanted", "60 s in total would require
# more footage", "there isn't enough footage for 60 s total", "closer to 60 s overall", "a total of
# 60 s isn't reachable yet"). Unsure means no claim (#743).
_HEDGE = re.compile(
    r"\b(?:not|never|cannot|no\s+(?:more|further|way)|would|could|might|if|unless|enough"
    r"|possible|impossible|you|ask(?:ed)?|want(?:ed)?|request(?:ed)?|target|goal|reach\w*"
    r"|need\w*|requir\w*|toward|towards|closer|below|under|(?:short|shy)\s+of)\b|n['\u2019]t\b",
    re.I,
)
_THE = re.compile(r"\bthe\s+$", re.I)
_BY = re.compile(
    r"\bby\s+(?:(?:about|around|roughly|approx\w*|nearly|almost|just)\s+|~\s*)?$", re.I
)
# "4 min 30 s overall": the minutes before the seconds
_MINUTES_BEFORE = re.compile(rf"(?<![\w.,]){_NUM}\s*-?\s*(?:minutes?|mins?)\s+(?:and\s+)?$", re.I)


def _claim_value(m: re.Match[str]) -> float:
    if m.group("mm") is not None:
        return float(int(m.group("mm")) * 60 + int(m.group("ss")))
    n = _count(m.group("n"))
    if not m.group("u").lower().startswith("m"):
        if minutes := _MINUTES_BEFORE.search(m.string, max(0, m.start("n") - 40), m.start("n")):
            return _count(minutes.group(1)) * 60 + n
        return n
    more = _AND_SECONDS.match(m.string, m.end())
    if more and more.group(2).lower().startswith("s"):
        return n * 60 + _count(more.group(1))
    return n * 60


def claimed_seconds(summary: str) -> float | None:
    """The length in seconds that a proposal's ``summary`` gives the whole sequence, the last one
    tied to a cue ("now about 34 s", "a total of 90 s", "the video is now 90 seconds", "now runs
    1:30", "90 s in all"), else None. A bound ("under 1 minute"), a length without a cue ("added
    about 12 s") and a clip's length ("the jump is now 8 s", "m3 now runs 12 s") are no claim.
    Unlike :func:`target_seconds`, which reads what the user asks for."""
    found: list[tuple[int, float]] = []
    for m in _CLAIM_AFTER.finditer(summary):
        if not _other_length(m):
            found.append((m.start(), _claim_value(m)))
    for m in _CLAIM_BEFORE.finditer(summary):
        if _other_length(m):
            continue
        if m.group("cue").lower() in _SUBJECT_CUES:
            clause = _CLAUSE.search(summary, 0, m.end("cue")).group()  # type: ignore[union-attr]
            sentence = _SENTENCE.search(summary, 0, m.start()).group()  # type: ignore[union-attr]
            if (
                _OTHER_SUBJECT.search(clause)
                or _NOUN_NOW.search(clause)
                or (_CLIP_CHANGE.search(sentence) and not _VIDEO_SUBJECT.search(clause))
            ):
                continue
        found.append((m.start(), _claim_value(m)))
    return max(found)[1] if found else None


_NAMED_LEN = re.compile(rf"(?<![\w:.]){_CLAIM_LEN}", re.I)
# Before a length: the length before a change ("from 34 s to 60 s", "up from 0:34"). Other
# lengths stay: "at 34 s", "the 34 s cut", "(now 34 s)" and "34 s more or less" can be the
# sequence's, and a correct summary is never flagged for a length left out.
_FROM = re.compile(
    r"\bfrom\s+(?:(?:about|around|roughly|approx\w*|nearly|almost|just)\s+|~\s*)?$", re.I
)


def named_seconds(summary: str) -> list[float]:
    """Every length that ``summary`` names, tied to a cue or not ("now 34 s", "60 s, which you
    asked for", "1:30"), in seconds; not the length before a change ("from 34 s to 60 s")."""
    out = []
    for m in _NAMED_LEN.finditer(summary):
        at = m.start("n") if m.group("n") is not None else m.start("mm")
        if not _FROM.search(summary, max(0, at - 40), at):
            out.append(_claim_value(m))
    return out


def _other_length(m: re.Match[str]) -> bool:
    """Whether the length of a claim match is another one than the sequence's: in parentheses
    ("(12 s in all)"), a change or a clip's ("12 s longer", "5 s each"), a place in the source
    ("12:30 in the source"), the object of a change ("Cut 20 s in total", "Removed a total of
    20 s") or a length asked for, negated or hypothetical ("You asked for a total of 60 s", "not
    the 60 s overall you wanted", "60 s, which you asked for", "60 s total isn't reachable")."""
    text = m.string
    at = m.start("n") if m.group("n") is not None else m.start("mm")
    head = text[max(0, at - _LOOKBACK) : at]
    if head.rfind("(") > head.rfind(")") or _NOT_SEQ_AFTER.match(text, m.end()):
        return True
    part = _PART.search(head).group()  # type: ignore[union-attr]
    after = _PART_AFTER.match(text, m.end()).group()  # type: ignore[union-attr]
    nxt = _NEXT_PART.match(text, m.end() + len(after))
    if _THE.search(part) or any(_HEDGE.search(s) for s in (part, after, nxt[1] if nxt else "")):
        return True
    if _BY.search(part):  # "Trimmed it by 20 s overall": a change
        return True
    verbs = list(_CHANGE_VERB.finditer(part))
    return bool(verbs) and not _CHANGE_TO.search(part, verbs[-1].end())


# A text reply that says more work follows ("Now let me search for ...") without doing it, and a
# message that asks for an edit: such a reply gets the one nudge to go on with the tools.
ANNOUNCES = re.compile(
    r"(?:^|[.!?:\n])\s*(?:(?:now|ok|okay|next|first|so|alright)\b,?\s*)?(?:let me|let's|i'll|i will"
    r"|i am going to|i'm going to|i need to|i should)\s+(?:now\s+)?(?:search|find|look|add|check"
    r"|extend|build|get|propose|put|trim|lengthen|pick|select|create|make|fetch|query|use|grab)\b",
    re.I,
)
# A message that asks for an edit: a turn that ends in text gets the one nudge toward a proposal.
# A question that asks for advice ("what would you trim?", "anything to fix?") asks none.
_ADVICE = re.compile(
    r"\s*(?:what|which|why|where|when|who|how\b(?!\s+about)|is|are|does|do|should|any|anything)\b",
    re.I,
)
_EDIT_ASK = re.compile(
    r"\b(?:make|build|add|put|place|change|extend|lengthen|trim|shorten|remove|delete|move|reorder"
    r"|sort|create|assemble|cut|fix|longer|shorter|arrange|include|insert|swap|replace|color|grade"
    r"|stitch|splice)\b",
    re.I,
)
# Edit words that are common in help questions too ("how do I use the timeline?", "can it use my
# GPU?"): an edit only where they open a clause or follow a request word ("drop the intro", "ok,
# use the laugh", "can you take out the menu part?", "let's start with the crash"). "use" after
# "you" or "we" only with a moment or a time later in the sentence ("can you use m3?", "can you use
# the laugh at 12:30?"; not "can you use my GPU?", "we use OBS to record.")
_EDIT_LEAD = re.compile(
    r"(?:(?:^\s*|[.!;,]\s*|\b(?:please|just|then|and|also|you|we|let's)\s+)"
    r"(?:drop|get rid|take out|(?:open|start|end)\s+with)"
    r"|(?:^\s*|[.!;,]\s*|\b(?:please|just|then|and|also|let's)\s+)use"
    r"|\b(?:you|we)\s+use(?=[^.?!]*\b(?:m\d+|\d+:\d\d)\b))\b",
    re.I,
)
_LAST_SENTENCE = re.compile(r"[^.:;!]*$")
# The last sentence of a reply that promises a lookup next ("I'll search each one now.", "Let me
# check the transcript"); not one that offers it ("I'll search more if you want") nor one with
# another verb ("Let me know if ...", "I'll leave it there", "Let's call it a draw")
_PROMISES = re.compile(
    r"(?:(?:now|ok|okay|next|so|alright|then)\b,?\s*)?(?:let's\s+(?:now\s+|also\s+|first\s+"
    r"|next\s+)?(?:search|find|look|check|scan|query|fetch)|(?:let me|i'll|i will|i need to"
    r"|i am going to|i'm going to|i'm gonna)\s+(?:now\s+|also\s+|first\s+|next\s+)?(?:search"
    r"|find|look|check|scan|query|fetch|get(?!\s+back)|view|watch|run|try|go through|do that"
    r"|do it))\b",
    re.I,
)
# ... or one that is only a moment asked for ("One moment", "Give me a second"; not "One moment
# stands out: ...", "A moment later, she laughs")
_MOMENT_ONLY = re.compile(
    r"(?:(?:please\s+)?give me|just)?\s*(?:one|a)\s+(?:moment|sec(?:ond)?)", re.I
)
# ... but not one that gives its answer after a colon: a time, a moment id, a quoted word or a
# clause ("Let me check: she says it at 12:30", "Let's look at it another way: the jump is the
# best part"); a list of what is looked for next stays a promise ("Let me check for the other
# terms: grifter, loser, and curse words")
_ANSWERED = re.compile(
    r"\d|\bm\d+\b|\"|(?<!\w)'|'(?!\w)|\b(?:is|are|was|were|says|said|has|have|had|appears?"
    r"|shows?|does|did|no|not|none|nothing|never|only)\b|n't\b",
    re.I,
)
# ... or one that opens with a gerund, only when it ends in "now" or the reply in an ellipsis
# ("Searching now.", "Checking the transcript..."; not "Looking at the transcript, she never
# says it.")
_GERUND = re.compile(r"(?:searching|looking|checking|scanning)\b", re.I)
_NOW = re.compile(r"\bnow$", re.I)
_COLON = re.compile(r":\s+")
_OFFERS = re.compile(
    r"\b(?:if you|would you|do you|want me|you want|you'd like|you would like|next time|later)\b"
    r"|(?<!see )(?<!check )\bif\b",
    re.I,
)
_SENTENCES = re.compile(r"[.!?]+\s+|\n+")


def promises(text: str) -> bool:
    """Whether a reply ends in a promise of a lookup (:data:`_PROMISES`), or in a colon with
    nothing after it in any language: work announced and not done. A question is no promise."""
    text = re.sub(r"\u2019", "'", text).rstrip()
    trails = text.endswith(("...", "\u2026"))  # an ellipsis: more follows
    text = text.rstrip(".!\u2026 ")
    if text.endswith(":"):
        return True
    if text.endswith("?"):
        return False
    last = _SENTENCES.split(text)[-1].strip().lstrip("*_#>- ")
    if _MOMENT_ONLY.fullmatch(last):
        return True
    head, after = last, ""
    if colon := _COLON.search(last):  # a colon before a space, not one in a time (12:30)
        head, after = last[: colon.start()], last[colon.end() :]
    if _OFFERS.search(last) or _ANSWERED.search(after):
        return False
    if _GERUND.match(head):  # "Looking at the rest, nothing else." answers
        if _ANSWERED.search(head.partition(",")[2]):
            return False
        return trails or bool(_NOW.search(last))
    return bool(_PROMISES.match(head))


def edits(text: str) -> bool:
    """Whether ``text`` has an edit word (:data:`_EDIT_ASK`, :data:`_EDIT_LEAD`)."""
    return bool(_EDIT_ASK.search(text) or _EDIT_LEAD.search(text))


def asks_advice(text: str) -> bool:
    """Whether a message is a question that asks for advice, not for an edit: it ends in "?"
    and opens with an advice word or has no edit word ("what would you trim?", "should I drop
    the intro?", "how do I use the timeline?", "can it be 40 s long?"). "could we add the
    laugh?" asks for an edit. With only a leading edit word, a last sentence that opens with an
    advice word asks for advice too ("Let's start with the basics: what is a style?"). One rule
    for the proposal nudge and the auto-fill offer."""
    text = text.rstrip()
    if not text.endswith("?"):
        return False
    if _ADVICE.match(text) or _EDIT_ASK.search(text):
        return bool(_ADVICE.match(text))
    if not _EDIT_LEAD.search(text):
        return True
    return bool(_ADVICE.match(_LAST_SENTENCE.search(text[:-1]).group()))  # type: ignore[union-attr]


_REMOVES = re.compile(
    r"\b(?:remove|delete|drop|clear|get rid|discard|take out|cut out|replace|swap|start over"
    r"|from scratch)\b",
    re.I,
)


def wants_removal(text: str) -> bool:
    """Whether a message asks to take clips out (so a proposal may drop half of them)."""
    return bool(_REMOVES.search(text) or CLEARS.search(text))


# A message that asks to empty the sequence: only then may a proposal remove every clip.
CLEARS = re.compile(
    r"\b(?:start (?:over|again|fresh|from scratch)|from scratch|begin again"
    r"|(?:clear|empty)(?: out)? (?:the |my )?(?:whole |entire )?(?:sequence|timeline)"
    r"|(?:remove|delete|drop|clear)\s+(?:everything|all (?:of )?(?:the |my )?clips?"
    r"(?=\s*(?:$|[.,;!]|and\b|from\b|in\b))))",
    re.I,
)


_NOT_AUDIO = (
    r"(?!\s+(?:(?:some|the|a|my|any|more|less|background|extra) )*"
    r"(?:music|audio|sound|songs?|tracks?|volume|voice|voiceover|sfx|levels?))"
)
# "between the runs", "between the <any word> runs", "between the action clips" and "in between
# the action": no activity is named, so a request reads the same for any footage (#1085, A303).
# A bare "between the action" is no mix ("trim the dead time between the action").
_BETWEEN = r"action (?:runs?|shots?|clips?)|(?:\w+ )?runs?"
_MIX = re.compile(
    r"(?<![\w-])(?:"
    r"break(?:s|ing)? (?:it )?up(?!(?:\W+\w+){0,4}?\W+into\b)|break(?:s|ing)? the (?:monotony|runs?)"
    rf"|mix(?:ed|es|ing)? (?:in|of|with){_NOT_AUDIO}|mixture of"
    r"|inters(?:perse|persed|perses|persing)|sprinkl(?:e|es|ed|ing)|scatter(?:s|ed|ing)?"
    r"|occasional(?:ly)?(?! (?:stutter|glitch|flicker|jitter|artifact|noise|blur|drop))"
    r"|now and (?:then|again)|from time to time"
    rf"|in[- ]between (?:the |some |all |each )?(?:{_BETWEEN}|action)"
    rf"|between (?:the |each |every |some )?(?:{_BETWEEN})"
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


_CLAUSE_BREAK = re.compile(r"[,;:.!?]|(?<![\w-])(?:but|just|instead|rather|then)(?![\w-])", re.I)


def _wanted(pattern: re.Pattern[str], text: str, *, transitions: bool = False) -> bool:
    """Whether ``pattern`` matches ``text`` somewhere that the words before it do not negate
    (or, with ``transitions``, turn into a fade or cut)."""
    for m in pattern.finditer(text):
        # only the clause the match is in: a "no" or "fade" before a comma or "but" is not about it
        before = _CLAUSE_BREAK.split(text[max(0, m.start() - 40) : m.start()])[-1]
        if not _NEG.search(before) and not (transitions and _TRANSITION.search(before)):
            return True
    return False


def asks_for_mix(text: str) -> bool:
    """Whether the message asks to break up, mix in, intersperse or alternate kinds ("an
    occasional funny moment to break up the action"). An audio mix ("mix the audio", "no music
    mix"), a transition between clips, and a request that is negated ("don't break up the
    action") are none."""
    return _wanted(_MIX, text, transitions=True)


def names_dialogue(text: str) -> bool:
    """Whether the message asks for dialogue (dialogue, conversation, banter, commentary, talk or
    chat moments), unless the words before it negate it ("no dialogue")."""
    return _wanted(_DIALOGUE, text)
