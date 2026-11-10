"""Registry of named scoring features: what each one measures, its layer and the extra it needs.

Every style weighs features registered here (:mod:`cutpoint.styles` checks each weight name).
A ``grid`` feature is a 0-1 signal on the 0.1 s analysis grid; the weighted sum of the grids is
peak-picked into candidates (:mod:`cutpoint.scorer`). A ``bonus`` feature adds points to a
candidate's score afterwards (the events rescore). A ``part`` records what a grid feature is
built from (action-cam's ``social``: ``social_said`` and ``social_heard``); it cannot be weighted, is
not listed as available and adds no grid of its own.

A feature is *available* when the extra it needs is installed. On Cutpoint lite (no ``events``
extra) the features of that extra never run, so the score divides by the weights of the
available features only: a 0-100 score then means the same on lite and full installs, and such
a renormalised score is capped at 100 (a guard for grids cached by a full install and for
learned presets). With every feature available (a full install, or a style of core features
only) the divisor is the sum of all weights and nothing is capped, as before the registry.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from cutpoint import extras, state

logger = logging.getLogger(__name__)

LAYERS = ("audio", "transcript", "telemetry", "events", "visual")
KINDS = ("grid", "bonus", "part")


@dataclass(frozen=True)
class Feature:
    """One named scoring feature.

    Attributes:
        name: Registry name, as styles' ``weights`` use it.
        layer: Where it comes from: ``audio``, ``transcript``, ``telemetry``, ``events`` or
            ``visual``.
        description: What it measures, in plain language.
        extra: The optional extra it needs (``"events"``), or None for core.
        kind: ``"grid"`` (0-1 on the analysis grid, weighted and summed), ``"bonus"``
            (points added to a candidate's score) or ``"part"`` (one source of a grid
            feature's ``parts``: not weighted, not available on its own, no grid).
        parts: The ``part`` features a grid feature is built from, if it records them. The
            feature's own ``extra`` says where it runs: a core feature can have a part that
            needs an extra (it is 0 without it).
    """

    name: str
    layer: str
    description: str
    extra: str | None = None
    kind: str = "grid"
    parts: tuple[str, ...] = ()


REGISTRY: dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    """Add ``feature`` to :data:`REGISTRY` and return it.

    Raises:
        ValueError: For an unknown layer or kind, another feature of the same name, or a
            ``parts`` name that is not a registered ``part`` feature.
    """
    if feature.layer not in LAYERS:
        raise ValueError(f"{feature.name}: unknown layer {feature.layer!r}; choose from {LAYERS}")
    if feature.kind not in KINDS:
        raise ValueError(f"{feature.name}: unknown kind {feature.kind!r}; choose from {KINDS}")
    bad = [p for p in feature.parts if p not in REGISTRY or REGISTRY[p].kind != "part"]
    if bad:
        raise ValueError(f"{feature.name}: parts {bad} are not registered part features")
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


@state.register_cache
@functools.cache
def installed_extras() -> frozenset[str]:
    """The feature extras installed here, detected once per run: ``{"events"}`` in the full
    edition, none in lite (:func:`cutpoint.extras.edition`)."""
    return frozenset({"events"}) if extras.edition() == "full" else frozenset()


def available(installed: Collection[str] | None = None) -> frozenset[str]:
    """Names of the features whose extra is installed (core features always); never a
    ``part``, which is not a feature of its own.

    Args:
        installed: The installed extras; default :func:`installed_extras`. Tests pass
            ``()`` for lite or ``("events",)`` for full.
    """
    have = installed_extras() if installed is None else frozenset(installed)
    return frozenset(
        n for n, f in REGISTRY.items() if f.kind != "part" and (f.extra is None or f.extra in have)
    )


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
    grids: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    order: Iterable[str],
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted sum of ``grids`` over the names in ``order`` (a missing weight is 0).

    The caller passes the term order: floating-point addition is not associative, so the same
    order gives the same bytes. Zero-weight terms are added too, as before.

    ``start`` is what the terms are added to. None is ``sum()``'s int 0, as the generic styles
    summed before the registry: the sum keeps the grids' dtype. Action-cam passes float64 zeros, as it
    did: its cached telemetry grids are float32, and each float32 term is then added in float64.
    """
    total: np.ndarray = 0 if start is None else start  # type: ignore[assignment]  # 0 + array
    for k in order:
        total = total + weights.get(k, 0.0) * grids[k]
    return total


def dropped(weights: Mapping[str, float], avail: Collection[str] | None) -> list[str]:
    """The weights left out of :func:`divisor`: registered grid features not in ``avail``
    (none for ``avail`` None or a full install), in ``weights`` order."""
    if avail is None:
        return []
    grid = {n for n, f in REGISTRY.items() if f.kind == "grid"}
    return [k for k in weights if k in grid and k not in avail]


def to_score(peak: float, weights: Mapping[str, float], avail: Collection[str] | None) -> float:
    """A combined-grid peak as a 0-100 score (one decimal), renormalised over ``avail``
    (:func:`divisor`).

    A renormalised score is capped at 100. A built-in style analysed on lite stays within it:
    every grid is 0-1 and the grids of left-out features are 0, so the peak is at most the
    divisor. Action-cam's ``social`` counts in that divisor: it is core, as its ``social_said`` part
    (transcript laughter) runs on lite, and only its ``social_heard`` part (sound events) is 0
    there. The cap guards what can still pass it: grids and events a full install cached in
    a shared project (the voice and visual layers, ``events.json``) and learned presets. With
    nothing left out (every full install) the score is the plain quotient, as before the
    registry.
    """
    score = round(100 * (peak / divisor(weights, avail)), 1)
    return min(100.0, score) if dropped(weights, avail) else score


# --------------------------------------------------------------------------- generic features
for _f in (
    Feature("loudness", "audio", "How much louder the moment is than the recording usually is"),
    Feature("speech", "transcript", "The style's keywords and exclamations being said"),
    Feature("laughter", "transcript", "Laughter written in the transcript (haha, lol)"),
    Feature("continuity", "transcript", "How much of the surrounding ten seconds is talk"),
    Feature("scene", "visual", "How often the picture cuts to a new shot"),
):
    register(_f)

# --------------------------------------------------------------------------- action-cam features
# The parts of action-cam's social grid (#556): cutpoint.actioncam takes the larger of the two. social is
# core, so it counts in the lite divisor: there only social_said runs.
for _f in (
    Feature("social_said", "transcript", "Laughter written in the transcript", kind="part"),
    Feature(
        "social_heard",
        "events",
        "Laughter, screams or cheering heard (PANNs sound events)",
        "events",
        "part",
    ),
):
    register(_f)
# cutpoint.actioncam builds these grids; the scorer weighs them like the generic ones (#514 phase 1b).
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
    Feature(
        "social",
        "events",
        "Laughter, screams or cheering heard, or laughter said",
        parts=("social_said", "social_heard"),
    ),
    Feature("voice_laughter", "audio", "Laughter heard over the wind noise", "events"),
    Feature("voice", "audio", "Someone talking over the wind noise", "events"),
    Feature("beauty", "visual", "A steady, good-looking view", "events"),
    Feature("people", "visual", "Friends close to the camera", "events"),
    Feature("group", "visual", "A group of people in view", "events"),
):
    register(_f)
# The action-cam grids in cutpoint.actioncam's order: the order of the saved action_cam_features arrays and of a
# candidate's signals, and the first of equal contributions is a pick's kind.
ACTION_CAM_FEATURES = (
    "airtime",
    "crash",
    "vlm",
    "speed",
    "social",
    "carve",
    "scenic",
    "speech",
    "faces",
    "stopped",
    "short_clip",
    "beauty",
    "people",
    "group",
    "voice_laughter",
    "voice",
    "wipeout",
)
# Before #514 phase 1b the action-cam feature voice_laughter was called "laughter", which is the
# transcript feature for the generic styles. Learned action-cam presets and saved feature files of
# that time say "laughter"; action_cam_names() reads it as voice_laughter.
ACTION_CAM_ALIASES = {"laughter": "voice_laughter"}
# Features that no longer exist. A learned preset saved while they did may still weigh them;
# styles.load_learned() drops those weights, so they are not counted in the divisor and the
# preset scores as one without them. ``taste`` was the personal taste head
# (``refmatch --taste``), removed when refmatch became scripts/refmatch.py.
REMOVED = frozenset({"taste"})


def action_cam_names[T](named: Mapping[str, T], where: str = "") -> dict[str, T]:
    """``named`` (action-cam weights or grids) with the old names (:data:`ACTION_CAM_ALIASES`) renamed, in
    place of the old key (the order, and so a score's sum, stay the same). A name given both
    ways keeps the new name's value. ``where`` names the file in the DEBUG line."""
    out: dict[str, T] = {}
    for k, w in named.items():
        new = ACTION_CAM_ALIASES.get(k)
        if new is None:
            out[k] = w
            continue
        logger.debug("%s: action-cam feature %r is now called %r", where or "weights", k, new)
        if new not in named:
            out[new] = w
    return out


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
