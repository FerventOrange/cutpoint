"""``preview_ranking``: the project's whole moments list re-ranked in another style (#902).

``find_moments`` and ``find_words`` look particular moments up. A request about the whole list
("show me the funny bits first", "more talk") asks for another ranking instead: the moments
scored again in a library style (built in, or one the user saved) from the analysis caches,
without analyzing again. The tool only previews it: the page shows the before/after card of
``ui.ranking_preview`` and the user applies it with Use this ranking (or analyzes again, when
the caches cannot give that style). Nothing here writes project files.

The page offers the tool through a :class:`Hook` (``ui.RankingHook``): the styles to choose
from, with a note each (:func:`listed`: :data:`STYLE_NOTES` for a built-in, the saved style's
description for the user's own), the style the moments are ranked in now, and the preview. A
project without one (no analysis, a 360 project) is not offered the tool. A prompt at its floor
(a small context) gets the :func:`brief` rules instead of :func:`rules`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

NAME = "preview_ranking"
JSON_TOOL = f"{NAME} {{style, reason?}}"
MAX_STYLES = 30  # styles offered (built-ins first, as the library lists them)
NOTE_CHARS = 60  # of a saved style's description
MAX_REASON = 200
MAX_RESULT = 800  # characters of the result text
TOP_LABELS = 3  # labels the result names
LABEL_CHARS = 40
# One short note per built-in style (docs/configuration.md, Styles)
STYLE_NOTES = {
    "gaming": "loudness and hype words",
    "action": "loudness and scene cuts, plus laughter",
    "sport": "crowd loudness, goal calls, laughter",
    "funny": "laughter first",
    "commentary": "long talk, speech continuity, in time order",
    "action-cam": "camera telemetry: speed, airtime, crashes",
}


class Hook(Protocol):
    """What the page gives a turn to preview rankings with."""

    def styles(self) -> list[tuple[str, str]]:
        """The library styles as ``(name, description)``; "" for a built-in."""
        ...

    def current(self) -> str:
        """The style the project's moments are ranked in now."""
        ...

    def preview(self, name: str) -> dict[str, Any]:
        """``ui.ranking_preview`` of the style ``name`` (its ready or not-ready shape).

        Raises:
            RankingError: The preview was refused.
        """
        ...


class RankingError(Exception):
    """A preview refused (``busy``: a run or a ranking holds the project)."""

    def __init__(self, message: str, busy: bool = False) -> None:
        super().__init__(message)
        self.busy = busy


def listed(styles: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """``styles`` as the model sees them: ``(name, note)``, the note of a built-in from
    :data:`STYLE_NOTES`, else the description cut to :data:`NOTE_CHARS`; at most
    :data:`MAX_STYLES`."""
    out = []
    for name, description in styles:
        note = STYLE_NOTES.get(name) or " ".join(str(description or "").split())
        if len(note) > NOTE_CHARS:
            note = note[: NOTE_CHARS - 1].rstrip() + "…"
        out.append((str(name), note))
    return out[:MAX_STYLES]


def tool(styles: list[str]) -> dict[str, Any]:
    """The tool definition with this turn's ``styles`` as the enum of ``style``."""
    return {
        "type": "function",
        "function": {
            "name": NAME,
            "description": (
                "Preview the whole moments list ranked in another style, from the cached "
                "analysis: the user sees the new top moments on a card and decides with Use "
                "this ranking (nothing changes until then). For asks about the whole list "
                "('show me the funny bits first', 'more talk'); not for finding moments. "
                "reason: one short sentence for the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "enum": list(styles)},
                    "reason": {"type": "string"},
                },
                "required": ["style"],
            },
        },
    }


RULES = f"""

{NAME}(style): when the user wants the whole list ranked differently ("show me the funny \
bits first", "more talk", "rank it for sport"), call {NAME} with the style that fits. For \
particular moments ("where does she laugh?") use find_moments or find_words instead. It only \
shows a preview card: the user clicks Use this ranking to apply it, so never say a ranking was \
applied. The moments are ranked as {{current}} now. Styles: {{styles}}."""


def rules(styles: list[tuple[str, str]], current: str) -> str:
    """:data:`RULES` with the styles (``(name, note)``, :func:`listed`) and the current one."""
    shown = "; ".join(f"{n} ({note})" if note else n for n, note in styles)
    return RULES.replace("{current}", current or "unknown").replace("{styles}", shown)


# The rules in a prompt at its floor (``context.Budget.chars``, a small context such as 8k
# tokens), where every character of rules is one the moment table loses: the behaviour, the
# current style and the notes of the user's own styles only (a built-in's name says enough).
# The tool's enum names the styles; JSON mode, which has no enum, gets every name.
BRIEF_RULES = f"""

{NAME}: the whole list ranked another way; find_moments for particular moments. A preview: \
never say it was applied. It is ranked as {{current}} now.{{styles}}"""


def brief(styles: list[tuple[str, str]], current: str, names: bool = False) -> str:
    """:data:`BRIEF_RULES` with the current style and the user's own styles of ``styles``
    (``(name, note)``, :func:`listed`) with their notes; every style's name with ``names``."""
    shown = [
        f"{n} ({note})" if note and n not in STYLE_NOTES else n
        for n, note in styles
        if names or (note and n not in STYLE_NOTES)
    ]
    text = f" Styles: {'; '.join(shown)}." if shown else ""
    return BRIEF_RULES.replace("{current}", current or "unknown").replace("{styles}", text)


def _new_in_top(payload: Mapping[str, Any]) -> tuple[int, int]:
    """``(moments new in the top, the top's length)`` of a ready preview."""
    top = [r for r in payload.get("top") or [] if isinstance(r, Mapping)]
    n = len(top)
    new = sum(1 for r in top if not isinstance(r.get("was"), int) or r["was"] > n)
    return new, n


def result(payload: Mapping[str, Any]) -> str:
    """The tool result for the model: what the card shows, in a few lines (file names and
    labels only, never a path)."""
    style, was = payload.get("style"), payload.get("from") or "unknown"
    if not payload.get("ready"):
        then = (
            "the user was offered Analyze again"
            if payload.get("can_analyze")
            else "the user was told to reconnect the footage"
        )
        why = str(payload.get("message") or "")[:400]
        return f"{style} needs a new analysis ({why}); {then}."[:MAX_RESULT]
    new, n = _new_in_top(payload)
    labels = []
    for r in (payload.get("top") or [])[:TOP_LABELS]:
        if isinstance(r, Mapping):
            labels.append(f"{r.get('rank')}. {str(r.get('label') or '')[:LABEL_CHARS]}")
    text = (
        f"Preview shown: {style} (was {was}), {new} new in the top {n}; top {len(labels)}: "
        f"{'; '.join(labels)}. The user decides with Use this ranking."
    )
    return text[:MAX_RESULT]


def reply(event: Mapping[str, Any]) -> str:
    """The turn's reply in ``chat.json`` (what later turns see) for a ranking event."""
    style = event.get("style")
    if not event.get("ready"):
        return f"Previewed the {style} ranking: it needs a new analysis; the user decides"
    new, n = _new_in_top(event)
    return f"Previewed the {style} ranking ({new} new in the top {n}); the user decides"
