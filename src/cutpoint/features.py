"""Registry of named scoring features: what each one measures, its layer and the extra it needs.

Every style weighs features registered here (:mod:`cutpoint.styles` checks each weight name).
A ``grid`` feature is a 0-1 signal on the 0.1 s analysis grid; the weighted sum of the grids is
peak-picked into candidates (:mod:`cutpoint.scorer`). A ``bonus`` feature adds points to a
candidate's score afterwards (the events rescore).

A feature is *available* when the extra it needs is installed. On Cutpoint lite (no ``events``
extra) the features of that extra never run, so the score divides by the weights of the
available features only: a 0-100 score then means the same on lite and full installs. With every
feature available (a full install, or a style of core features only) the divisor is the sum of
all weights, as before the registry.
"""

from __future__ import annotations

import functools
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from cutpoint import extras, state

LAYERS = ("audio", "transcript", "telemetry", "events", "visual")
KINDS = ("grid", "bonus")


@dataclass(frozen=True)
class Feature:
    """One named scoring feature.

    Attributes:
        name: Registry name, as styles' ``weights`` use it.
        layer: Where it comes from: ``audio``, ``transcript``, ``telemetry``, ``events`` or
            ``visual``.
        description: What it measures, in plain language.
        extra: The optional extra it needs (``"events"``), or None for core.
        kind: ``"grid"`` (0-1 on the analysis grid, weighted and summed) or ``"bonus"``
            (points added to a candidate's score).
    """

    name: str
    layer: str
    description: str
    extra: str | None = None
    kind: str = "grid"


REGISTRY: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    """Add ``feature`` to :data:`REGISTRY` and return it.

    Raises:
        ValueError: For an unknown layer or kind, or another feature of the same name.
    """
    if feature.layer not in LAYERS:
        raise ValueError(f"{feature.name}: unknown layer {feature.layer!r}; choose from {LAYERS}")
    if feature.kind not in KINDS:
        raise ValueError(f"{feature.name}: unknown kind {feature.kind!r}; choose from {KINDS}")
    old = REGISTRY.get(feature.name)
    if old is not None and old != feature:
        raise ValueError(f"A different feature named {feature.name!r} is already registered")
    REGISTRY[feature.name] = feature
    return feature


def get(name: str) -> Feature:
    """The feature ``name``.

    Raises:
        ValueError: If no feature has that name.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown feature {name!r}; choose from {names()}") from None


def names(kind: str | None = None) -> list[str]:
    """Registered feature names (of ``kind`` only, if given), sorted."""
    return sorted(n for n, f in REGISTRY.items() if kind is None or f.kind == kind)


def unknown(weights: Iterable[str]) -> list[str]:
    """The names in ``weights`` that are not registered, in their order."""
    return [k for k in weights if k not in REGISTRY]


@state.register_cache
@functools.cache
def installed_extras() -> frozenset[str]:
    """The feature extras installed here, detected once per run: ``{"events"}`` in the full
    edition, none in lite (:func:`cutpoint.extras.edition`)."""
    return frozenset({"events"}) if extras.edition() == "full" else frozenset()


def available(installed: Collection[str] | None = None) -> frozenset[str]:
    """Names of the features whose extra is installed (core features always).

    Args:
        installed: The installed extras; default :func:`installed_extras`. Tests pass
            ``()`` for lite or ``("events",)`` for full.
    """
    have = installed_extras() if installed is None else frozenset(installed)
    return frozenset(n for n, f in REGISTRY.items() if f.extra is None or f.extra in have)


def divisor(weights: Mapping[str, float], avail: Collection[str] | None = None) -> float:
    """Sum of the weights of the available features, in ``weights`` order (1.0 if it is 0).

    Zero weights are kept in the sum; ``avail`` None counts every weight. A name that is not a
    registered grid feature (in a hand-edited learned preset) always counts, as before the
    registry. Summing in the same order as before the registry keeps the floating-point result
    identical.
    """
    if avail is None:
        return sum(weights.values()) or 1.0
    grid = {n for n, f in REGISTRY.items() if f.kind == "grid"}
    return sum(w for k, w in weights.items() if k in avail or k not in grid) or 1.0


def combine(
    grids: Mapping[str, np.ndarray], weights: Mapping[str, float], order: Iterable[str]
) -> np.ndarray:
    """Weighted sum of ``grids`` over the names in ``order`` (a missing weight is 0).

    The caller passes the term order: floating-point addition is not associative, so the same
    order gives the same bytes. Zero-weight terms are added too, as before.
    """
    total: np.ndarray = 0  # type: ignore[assignment]  # sum()'s start: 0 + array is the array
    for k in order:
        total = total + weights.get(k, 0.0) * grids[k]
    return total


def to_score(peak: float, weights: Mapping[str, float], avail: Collection[str] | None) -> float:
    """A combined-grid peak as a 0-100 score (one decimal), renormalised over ``avail``
    (:func:`divisor`)."""
    return round(100 * (peak / divisor(weights, avail)), 1)


# --------------------------------------------------------------------------- generic features
for _f in (
    Feature("loudness", "audio", "How much louder the moment is than the recording usually is"),
    Feature("speech", "transcript", "The style's keywords and exclamations being said"),
    Feature("laughter", "transcript", "Laughter written in the transcript (haha, lol)"),
    Feature("continuity", "transcript", "How much of the surrounding ten seconds is talk"),
    Feature("scene", "visual", "How often the picture cuts to a new shot"),
):
    register(_f)

# --------------------------------------------------------------------------- ski features
# Registered for the style check; the ski scorer (cutpoint.ski) still builds and weighs them.
for _f in (
    Feature("airtime", "telemetry", "Time in the air on a jump, from the camera's motion sensors"),
    Feature("crash", "telemetry", "A hard impact followed by a stop, from the motion sensors"),
    Feature("speed", "telemetry", "Going faster than usual for the day, from GPS or wind noise"),
    Feature("carve", "telemetry", "Rhythmic turns at speed, from the motion sensors"),
    Feature("faces", "telemetry", "Faces the camera detected while recording"),
    Feature("stopped", "telemetry", "Standing still"),
    Feature("short_clip", "telemetry", "A recording shorter than 30 s, made for one moment"),
    Feature("wipeout", "telemetry", "A crash found by cutpoint crashes, by its confidence"),
    Feature(
        "vlm", "events", "Jumps, landings, spray or friends seen by the vision model", "events"
    ),
    Feature("scenic", "events", "A scenic view, as the vision model tags it", "events"),
    Feature("social", "events", "Laughter, screams or cheering heard, or laughter said", "events"),
    Feature("voice_laughter", "audio", "Laughter heard over the wind noise", "events"),
    Feature("voice", "audio", "Someone talking over the wind noise", "events"),
    Feature("beauty", "visual", "A steady, good-looking view", "events"),
    Feature("people", "visual", "Friends close to the camera", "events"),
    Feature("group", "visual", "A group of people in view", "events"),
    Feature("taste", "visual", "Looks like the moments you picked in your own edit", "events"),
):
    register(_f)
# The ski scorer's "laughter" weight means voice_laughter, not the transcript feature above;
# phase 1b (#514) renames it in ski weights and keeps this alias for learned presets.
SKI_ALIASES = {"laughter": "voice_laughter"}

# --------------------------------------------------------------------------- bonus features
# The events rescore (cutpoint.scorer.rescore_events), applied in this order.
for _f in (
    Feature(
        "event_laughter",
        "events",
        "Laughter heard during the clip or just after it (up to 12 points)",
        "events",
        "bonus",
    ),
    Feature(
        "event_scream",
        "events",
        "A scream or cheering during the clip or just after it (up to 6 points)",
        "events",
        "bonus",
    ),
    Feature(
        "event_types",
        "events",
        "Game events on screen, such as kills and banners: points per event type",
        "events",
        "bonus",
    ),
    Feature(
        "event_reactions",
        "events",
        "A reaction (laughter, a scream, talk after a pause) right after an event, such as "
        "your own death in the funny style",
        "events",
        "bonus",
    ),
    Feature(
        "vlm_window",
        "events",
        "The vision model's rating of the moment: raises a lower score part of the way to it, "
        "never lowers it",
        "events",
        "bonus",
    ),
):
    register(_f)
