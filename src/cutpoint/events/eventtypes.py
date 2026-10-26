"""Event types: how OCR events are scored, extended and seeded, defined as data.

An event type matches OCR events (``when``) and says what they are worth:
``score`` points per event (at most ``max`` counted) and ``reaction`` points
once per candidate when a reaction follows one of its events, both in
:func:`cutpoint.events.scoring.event_boost` and both only for the project
styles listed in ``styles`` (unset: every style), whether the event is a payoff
for clip tails (``payoff``, ``tail_s``, :mod:`cutpoint.events.tails`) and its
priority as a VLM window seed (``seed``, :mod:`cutpoint.events.vlm_windows`).

HUD presets can define their own types in an ``events`` list::

    "events": [
      {"name": "objective", "when": [{"type": "banner", "prefix": ["OBJECTIVE CAPTURED"]}],
       "score": 6, "max": 2, "payoff": true, "tail_s": 2.5, "seed": 1.5,
       "reaction": 4, "styles": ["funny", "gaming"]}
    ]

A matcher takes the event ``type`` plus one of ``prefix`` / ``contains``
(upper-case label text) or ``flag`` (``player``, ``teamkill``,
``player_kill``, ...; several space-separated flags must all hold, ``!flag``
must be false). Preset types are checked before
:data:`DEFAULT_EVENT_TYPES` and replace a default with the same name; the
first matching type wins, whatever the project style (``styles`` gates only
the points). Unset ``payoff`` / ``tail_s`` / ``seed`` keep the
built-in rules, so the defaults reproduce the original behaviour exactly,
except that the player's own deaths (generic verb-style feeds and built-in R6
feeds alike) no longer count as player kills, a self-kill counts as
``self_kills`` (half a kill) instead, and in the ``funny`` style a death
followed by a reaction adds ``player_deaths``' reaction points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint.events.regions import HudPreset, as_list, game_key_for_path, load_preset
from cutpoint.styles import style_names

logger = logging.getLogger(__name__)

# Banner prefixes that count as payoffs (scoring "banners", clip tails).
BANNER_PAYOFFS = (
    "WON ROUND",
    "ROUND WON",
    "LOST ROUND",
    "ROUND LOST",
    "VICTORY",
    "DEFEAT",
    "ACE",
    "CLUTCH",
    "REVERSE FRIENDLY FIRE",
    "TARGET DESTROYED",
    "SHOT DOWN",
    "MISSION ACCOMPLISHED",
    # generic and verb-style feed banners
    "YOU HAVE BEEN DESTROYED",
    "CREW KNOCKED OUT",
    "MISSION FAILED",
    "MISSION COMPLETE",
    "YOU DIED",
    "DOUBLE KILL",
    "TRIPLE KILL",
    "MULTI KILL",
)
MAX_SCORE = 12.0  # custom types: points per event, and reaction points
MAX_COUNT = 3  # custom types: events counted per candidate
MAX_TAIL_S = 10.0  # custom types: reaction kept after a payoff, seconds
MAX_SEED = 3.0  # custom types: VLM seed priority (the built-in tiers top out at 2.0)
# Other evidence fields of ``signals.events`` (scoring.rescore_highlights); a type
# with one of these names would overwrite them. Names ending in ``_reaction`` are
# reserved too: ``<type>_reaction`` is the reaction evidence key of a type.
RESERVED_NAMES = frozenset({"laughter", "scream", "boost", "vlm"})


@dataclass(frozen=True)
class Matcher:
    """One ``when`` entry: an OCR event ``type`` plus an optional text or flag test."""

    type: str
    prefix: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    flag: str = ""

    def matches(self, e: dict[str, Any]) -> bool:
        """Whether an events.json entry or a raw layer-1 OCR event matches."""
        if e.get("kind") != "ocr" or e.get("type") != self.type:
            return False
        label = (e.get("label") or e.get("text") or "").upper()
        if self.prefix and not any(label.startswith(p) for p in self.prefix):
            return False
        if self.contains and not any(c in label for c in self.contains):
            return False
        # events.json keeps layer fields under "details"; layer-1 events have them flat
        flags = e["details"] if isinstance(e.get("details"), dict) else e
        return all(
            not flags.get(f[1:]) if f.startswith("!") else bool(flags.get(f))
            for f in self.flag.split()
        )


@dataclass(frozen=True)
class EventType:
    """A named class of OCR events and what it is worth.

    Attributes:
        name: Evidence key in ``signals.events`` (``player_kills``, ...).
        when: Matchers; any one matching is enough.
        score: Boost points per event (None: no boost, no evidence key).
        max: Events counted per candidate.
        reaction: Boost points added once per candidate when a reaction (laughter,
            a scream, or speech after a pause; see
            :func:`cutpoint.events.scoring.has_reaction`) follows one of its
            events (None: none, no ``<name>_reaction`` evidence key).
        styles: Project styles in which ``score`` and ``reaction`` apply (None:
            every style); a learned style counts by its name or its base.
            Matching ignores it.
        payoff: Clip-tail payoff (None: the built-in rule of :func:`tails.is_payoff`).
        tail_s: Reaction kept after the payoff (None: ``EventTailSettings.pad``).
        seed: VLM window seed priority (None: the built-in rule).
    """

    name: str
    when: tuple[Matcher, ...]
    score: float | None = None
    max: int = 1
    payoff: bool | None = None
    tail_s: float | None = None
    seed: float | None = None
    reaction: float | None = None
    styles: tuple[str, ...] | None = None

    def matches(self, e: dict[str, Any]) -> bool:
        """Whether any matcher matches the event."""
        return any(m.matches(e) for m in self.when)

    def applies(self, *names: str) -> bool:
        """Whether ``score`` and ``reaction`` count in a project whose style has one of
        ``names`` (its name and base); a type limited to some styles never counts
        without names."""
        return self.styles is None or any(n in self.styles for n in names)


# Order reproduces the original scoring chain: a teamkill (or an RFF banner)
# before the player's kills, then payoff banners and multikills. Banners are
# matched with startswith: their label is the matched phrase ("WON ROUND 5"),
# as in killfeed.link_teamkills. The player's own deaths (victim, not also
# killer) match "player_deaths" first, in every style: no per-event boost and
# no count key; in the funny style a reaction after the death (a friend's
# "look out" after it already happened) adds 8 points. The
# unset payoff / seed keep the built-in rules (a tail payoff, seed 1.2 in R6
# feeds, where ``player`` is set for either side). Self-kills (the player is
# killer and victim) come after teamkills and before both: their own type,
# worth half a kill, a tail payoff and a seed like a death. Player kills: kills
# by the player; ``player !verb`` still counts built-in rows that carry neither
# ``player_kill`` nor ``player_death``.
DEFAULT_EVENT_TYPES: tuple[EventType, ...] = (
    EventType(
        "teamkills",
        (Matcher("kill", flag="teamkill"), Matcher("banner", prefix=("REVERSE FRIENDLY FIRE",))),
        score=12.0,
        max=1,
    ),
    EventType(
        "self_kills",
        (Matcher("kill", flag="player_kill player_death"),),
        score=3.0,
        max=3,
    ),
    EventType(
        "player_deaths",
        (Matcher("kill", flag="player_death !player_kill"),),
        reaction=8.0,
        styles=("funny",),
    ),
    EventType(
        "player_kills",
        (Matcher("kill", flag="player_kill"), Matcher("kill", flag="player !verb")),
        score=6.0,
        max=3,
    ),
    EventType(
        "banners",
        (Matcher("banner", prefix=BANNER_PAYOFFS), Matcher("multikill")),
        score=6.0,
        max=2,
    ),
)


def _strs(v: Any) -> tuple[str, ...]:
    return tuple(str(x).upper() for x in as_list(v))


def _num(v: Any) -> float | None:
    return None if v is None or isinstance(v, bool) else float(v)


def _clamp(v: float | None, hi: float) -> float | None:
    return None if v is None else max(0.0, min(hi, v))


def _styles(v: Any) -> tuple[str, ...] | None:
    """``styles`` from preset JSON: a name or a list of names (empty: every style)."""
    names = tuple(str(x) for x in as_list(v))
    unknown = [n for n in names if n not in style_names()]
    if unknown:
        raise ValueError(f"unknown style(s) {unknown}; choose from {style_names()}")
    return names or None


def from_dict(data: dict[str, Any]) -> EventType:
    """An :class:`EventType` from preset JSON; ``score`` and ``reaction`` are clamped
    to 0-12, ``max`` to 1-3, ``tail_s`` to 0-10 and ``seed`` to 0-3.

    Raises:
        ValueError, TypeError, KeyError: When the entry is malformed (also: a name
            in :data:`RESERVED_NAMES`, a ``payoff`` that is not a JSON bool, a
            ``flag`` token that is a lone ``!`` or starts with ``!!``, a
            ``styles`` entry that is not a built-in or learned style).
    """
    when = tuple(
        Matcher(
            str(m["type"]),
            prefix=_strs(m.get("prefix")),
            contains=_strs(m.get("contains")),
            flag=str(m.get("flag") or ""),
        )
        for m in data["when"]
    )
    if not data.get("name") or not when:
        raise ValueError("an event type needs a name and at least one matcher")
    for m in when:
        if any(f == "!" or f.startswith("!!") for f in m.flag.split()):
            raise ValueError(f"bad flag {m.flag!r}")
    name = str(data["name"])
    if name in RESERVED_NAMES:
        raise ValueError(f"the name {name!r} is reserved")
    if name.endswith("_reaction"):
        raise ValueError(f"the name {name!r} is reserved (ends in _reaction)")
    payoff = data.get("payoff")
    if payoff is not None and not isinstance(payoff, bool):
        raise ValueError(f"payoff must be true or false, not {payoff!r}")
    return EventType(
        name,
        when,
        score=_clamp(_num(data.get("score")), MAX_SCORE),
        max=max(1, min(MAX_COUNT, int(data.get("max", 1)))),
        payoff=payoff,
        tail_s=_clamp(_num(data.get("tail_s")), MAX_TAIL_S),
        seed=_clamp(_num(data.get("seed")), MAX_SEED),
        reaction=_clamp(_num(data.get("reaction")), MAX_SCORE),
        styles=_styles(data.get("styles")),
    )


_warned_types: set[tuple[str, str]] = set()  # (path, game) of presets already warned about


def types_for(preset: HudPreset | None) -> tuple[EventType, ...]:
    """The preset's event types (invalid entries skipped) followed by the defaults it
    does not replace."""
    custom: list[EventType] = []
    warn_key = (str(preset.path), preset.game) if preset else None
    for d in preset.events if preset else ():
        try:
            custom.append(from_dict(d))
        except (ValueError, TypeError, KeyError, AttributeError) as err:
            if warn_key not in _warned_types:
                logger.warning(
                    "Ignoring invalid event type %r in HUD preset %s: %s", d, preset.game, err
                )
    if warn_key:
        _warned_types.add(warn_key)
    names = {t.name for t in custom}
    return (*custom, *(t for t in DEFAULT_EVENT_TYPES if t.name not in names))


def first_match(types: tuple[EventType, ...], e: dict[str, Any]) -> EventType | None:
    """The first type matching ``e`` (None if none does)."""
    return next((t for t in types if t.matches(e)), None)


def resolve(project: Path, ev_json: dict[str, Any]) -> dict[str, tuple[EventType, ...]]:
    """Event types per source of an events.json, from each source's HUD preset.

    The preset is looked up like the OCR layer did (``settings.ocr.game``, the
    project and the source path). Never raises: sources whose preset cannot be
    loaded get :data:`DEFAULT_EVENT_TYPES`.
    """
    game = (((ev_json.get("settings") or {}).get("ocr")) or {}).get("game")
    by_game: dict[str, tuple[EventType, ...]] = {}
    out: dict[str, tuple[EventType, ...]] = {}
    for e in ev_json.get("events", []):
        src = e.get("source")
        if src in out:
            continue
        try:
            key = game_key_for_path(str(src), game, project)
            if key not in by_game:
                by_game[key] = types_for(load_preset(key, project))
            out[src] = by_game[key]
        except Exception as err:  # noqa: BLE001 - a broken preset must not break scoring/clips
            logger.debug("Event types for %s: using the defaults (%s)", src, err)
            out[src] = DEFAULT_EVENT_TYPES
    return out
