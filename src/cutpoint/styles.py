"""Style presets controlling how highlight candidates are scored and cut."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from cutpoint import cache, features, paths, state

logger = logging.getLogger(__name__)


# Keyword patterns: a trailing "*" means prefix match ("fuck*" matches "fucking").
# Profanity is everyday filler in gaming voice chat. Styles with
# ``profanity_alone=False`` only count these when another cue is nearby.
PROFANITY = {"fuck*": 0.8, "shit*": 0.6, "damn*": 0.4}
_EXCLAIM = {"wtf": 1.0, "holy": 0.8}
_SWEARS = {**PROFANITY, **_EXCLAIM}
_HYPE = {
    "let's go": 1.5,
    "lets go": 1.5,
    "no way": 1.2,
    "oh my god": 1.2,
    "omg": 1.0,
    "wow": 0.8,
    "yes": 0.5,
    "insane": 1.0,
    "what": 0.3,
    "oh no": 0.8,
}
LAUGHTER = {
    "haha*": 1.0,
    "hah*": 0.8,
    "hehe*": 0.8,
    "lol": 0.8,
    "lmao*": 1.0,
    "dying": 0.6,
    "i'm dead": 0.8,
}


@dataclass(frozen=True)
class Style:
    """Weights and cut rules for one kind of video.

    Attributes:
        name: Preset name.
        weights: Weight per feature (registered in :mod:`cutpoint.features`;
            generic styles: loudness, speech, laughter, scene, continuity); the
            combined score is their weighted sum, normalised to 0-100 over the
            features this install can compute. ``continuity`` is the fraction
            of time someone is talking.
        vision_weight: The most a vision model rating can raise or lower a
            candidate's score, as a fraction, when vision is requested
            (``--vision``; on by default for every style): 0.3 means +30% at 10/10,
            -30% at 0/10 and no change at 5/10.
        pre_roll: Seconds of context kept before a peak.
        post_roll: Seconds kept after a peak.
        min_len: Minimum clip length in seconds.
        max_len: Maximum clip length in seconds.
        target_total: Suggested total highlight duration in seconds.
        merge_gap: Candidates closer than this (seconds) are merged.
        keywords: Phrase -> weight for the speech/excitement signal.
        snap_sentences: Snap boundaries to sentence/pause ends (long-form)
            instead of only avoiding mid-word cuts.
        default_order: Default ``cutpoint clips`` playback order.
        keyword_cap: If set, the summed keyword strength around any moment is
            capped at this value (strongest hit counts fully, the rest at 30%),
            so constant chatter cannot outrank one real exclamation.
        profanity_alone: If False, ``PROFANITY`` hits only count when a
            non-profane keyword, laughter or a loudness spike is within a few
            seconds, and they are never used as a clip label.
        smart_tail: Extend clips that end on speech past the last word until
            the audio (laughter, reaction, game sounds) settles; see
            ``cutpoint.tail``.
        tail_min: Guaranteed tail (s) after the last word of a speech-ended
            clip, unless an unrelated sentence starts sooner.
        tail_max: Maximum extension (s) after the last word.
        tail_settle: Seconds the level must stay near the local baseline for
            the reaction to count as over.
        speech_aware: Move clip boundaries that fall mid-utterance to the
            utterance's start/end (see ``cutpoint.boundaries``).
        speech_max_extend: Maximum seconds a speech-aware end moves later.
        speech_max_lead: Maximum seconds a speech-aware start moves earlier.
        kind: ``"generic"`` (loudness/speech/laughter/scene/continuity signals)
            or ``"action-cam"`` (telemetry + audio + VLM features, see
            :mod:`cutpoint.ski`; ``weights`` then name its features).
        use_scenes: Run scene detection by default (pointless on single-take
            action-cam clips, and the 4K proxy decode is the slowest step).
        learned: Fit metadata for presets loaded from the styles folder.
        mask_friends: Action-cam only: the lift/stop mask. Above 0 it is hard (no
            clips on lifts or stops; friends in view do not lift it, so every
            value above 0, such as the 5-11 of older presets, acts the same);
            0 = soft mask: nothing is forbidden and the ``stopped`` feature
            weight decides.
        open_list: The candidate list ends when moments stop qualifying (a stricter floor
            past :func:`cutpoint.analyze.usual_count` candidates), not at that count. Learned
            presets follow their base.
        must_include_crash: Crashes in ``crashes.json`` (``cutpoint crashes``, or the
            crash step of ``analyze``) with at least this confidence (0-1) become
            must-include moments, with their approach, aftermath and reaction, whatever
            the weights say (they replace the moments they overlap and rank first); 0 =
            off. Weaker crashes above the floor (:func:`cutpoint.crash.crash_floor`) are
            ranked like any moment (``analyze.add_crash_candidates``; action-cam: its
            ``wipeout`` feature). 0.6 for every style; learned presets may set
            ``"must_include_crash"``.
        file_hash: :func:`content_hash` of the style file a user style was loaded from
            (None for built-ins); see :func:`style_hash`.
    """

    name: str
    weights: dict[str, float]
    vision_weight: float
    pre_roll: float
    post_roll: float
    min_len: float
    max_len: float
    target_total: float
    merge_gap: float
    keywords: dict[str, float] = field(default_factory=dict)
    snap_sentences: bool = False
    default_order: str = "story"
    keyword_cap: float | None = None
    profanity_alone: bool = True
    smart_tail: bool = True
    tail_min: float = 1.5
    tail_max: float = 5.0
    tail_settle: float = 0.8
    speech_aware: bool = True
    speech_max_extend: float = 8.0
    speech_max_lead: float = 5.0
    kind: str = "generic"
    use_scenes: bool = True
    learned: dict | None = None
    mask_friends: float = 7.0
    must_include_crash: float = 0.6
    open_list: bool = False
    file_hash: str | None = None

    @property
    def base(self) -> str:
        """Built-in preset this style derives from (itself for built-ins)."""
        return (self.learned or {}).get("base", self.name)


_SKI_KEYWORDS = {
    **_HYPE,
    "woo*": 1.0,
    "yew": 1.0,
    "send it": 1.2,
    "sick": 0.8,
    "whoa": 1.0,
    "nice": 0.4,
    "yeah": 0.3,
    "dude": 0.4,
    "powder": 0.8,
    "pow": 0.6,
    "fell": 0.8,
    "crash*": 1.0,
    "wipe*": 1.0,
    "jump": 0.6,
    "air": 0.4,
}

# The telemetry-driven action-cam style (and its ``kind``).
ACTION_CAM = "action-cam"
# Old style names, no longer accepted: reading one is an error naming the new name.
RENAMED = {"ski": ACTION_CAM}
# ``--style auto`` (analyze, edit): the style follows the detected footage category
# (:mod:`cutpoint.category`). A reserved name: a learned ``auto.json`` is ignored.
AUTO = "auto"
ANALYZE_STYLE = "gaming"  # analyze's default --style (and --style auto's fallback)
# The style of each footage category per emphasis (analyze: hype; edit --kind
# funny-moments: funny). sport is never chosen automatically.
AUTO_STYLES: dict[str, dict[str, str]] = {
    "gameplay": {"hype": "gaming", "funny": "funny"},
    "action-cam": {"hype": ACTION_CAM, "funny": "funny"},
    "talk": {"hype": "commentary", "funny": "funny"},
    "general": {"hype": "action", "funny": "funny"},
}


@dataclass(frozen=True)
class AutoRequest:
    """``--style auto``: the style is detected (see :func:`resolve_auto`).

    Attributes:
        emphasis: ``hype`` or ``funny``.
        fallback: Style used when the detection is ambiguous (the command's default).
        must_include_crash: ``analyze --must-include-crash``, set on the style chosen.
    """

    emphasis: str
    fallback: str
    must_include_crash: float | None = None

    def style(self, name: str) -> Style:
        """The chosen style ``name``, with :attr:`must_include_crash` when given."""
        style = get_style(name)
        if self.must_include_crash is not None:
            style = replace(style, must_include_crash=self.must_include_crash)
        return style


def resolve_auto(category: str, emphasis: str) -> str:
    """The style name for a footage ``category`` (:data:`cutpoint.category.CATEGORIES`) and
    ``emphasis`` (``hype`` or ``funny``)."""
    return AUTO_STYLES[category][emphasis]


def renamed_error(name: object) -> str | None:
    """The error for a style ``name`` that was renamed (``ski``), else None."""
    new = RENAMED.get(name) if isinstance(name, str) else None
    if new is None:
        return None
    return (
        f"style {name!r} was renamed to {new!r}; use --style {new} "
        f"(a project analysed with {name!r} must be analysed again)"
    )


def style_arg(name: str) -> str:
    """``--style`` value check: a renamed style is refused with its new name."""
    if msg := renamed_error(name):
        raise argparse.ArgumentTypeError(msg)
    return name


def project_kind(hl: dict) -> str | None:
    """``style_kind`` of a highlights.json.

    Raises:
        ValueError: If the project names a renamed style (made before ``ski`` became
            ``action-cam``).
    """
    for key in ("style_kind", "style"):
        if msg := renamed_error(hl.get(key)):
            raise ValueError(msg)
    return hl.get("style_kind")


STYLES: dict[str, Style] = {
    "gaming": Style(
        name="gaming",
        weights={"loudness": 0.45, "speech": 0.35, "laughter": 0.15, "scene": 0.05},
        vision_weight=0.35,
        pre_roll=3.5,
        post_roll=2.5,
        min_len=4.0,
        max_len=12.0,
        target_total=90.0,
        merge_gap=3.0,
        keyword_cap=1.5,
        profanity_alone=False,
        open_list=True,
        keywords={
            **_HYPE,
            **_SWEARS,
            "clutch": 1.5,
            "ace": 1.2,
            "headshot": 1.0,
            "one tap": 1.0,
            "nice": 0.6,
            "dude": 0.4,
            "bro": 0.4,
            "sick": 0.6,
            "get him": 0.6,
            "he's dead": 0.8,
            "got him": 0.8,
            "kill*": 0.3,
            "help": 0.4,
        },
    ),
    "action": Style(
        name="action",
        weights={"loudness": 0.5, "speech": 0.2, "laughter": 0.1, "scene": 0.2},
        vision_weight=0.45,
        pre_roll=3.0,
        post_roll=3.0,
        min_len=3.0,
        max_len=12.0,
        target_total=60.0,
        merge_gap=2.0,
        keyword_cap=1.5,
        profanity_alone=False,
        keywords={**_HYPE, **_SWEARS, "whoa": 1.0, "go": 0.4, "jump": 0.6, "crash": 0.8},
    ),
    "sport": Style(
        name="sport",
        weights={"loudness": 0.5, "speech": 0.3, "laughter": 0.1, "scene": 0.2},
        vision_weight=0.4,
        pre_roll=7.0,
        post_roll=4.0,
        min_len=5.0,
        max_len=20.0,
        target_total=90.0,
        merge_gap=4.0,
        keywords={
            **_HYPE,
            "goal": 1.5,
            "score*": 1.0,
            "come on": 1.0,
            "what a": 1.2,
            "unbelievable": 1.2,
            "shot": 0.6,
            "save": 0.8,
            "touchdown": 1.5,
        },
    ),
    "funny": Style(
        name="funny",
        weights={"loudness": 0.3, "speech": 0.25, "laughter": 0.45, "scene": 0.0},
        vision_weight=0.3,
        pre_roll=5.0,
        post_roll=4.0,
        min_len=4.0,
        max_len=20.0,
        target_total=60.0,
        merge_gap=3.0,
        tail_min=1.8,
        tail_max=6.0,
        keywords={
            **_HYPE,
            **_SWEARS,
            "bro": 0.6,
            "dude": 0.6,
            "why": 0.5,
            "stop": 0.6,
            "i can't": 0.8,
            "what the": 0.8,
            "no no no": 1.0,
        },
    ),
    "commentary": Style(
        name="commentary",
        weights={
            "continuity": 0.45,
            "loudness": 0.3,
            "speech": 0.15,
            "laughter": 0.1,
            "scene": 0.0,
        },
        vision_weight=0.2,
        pre_roll=15.0,
        post_roll=15.0,
        min_len=20.0,
        max_len=120.0,
        target_total=750.0,
        merge_gap=8.0,
        keywords={
            **_HYPE,
            **_SWEARS,
            "clutch": 1.2,
            "ace": 1.0,
            "watch this": 1.0,
            "look at": 0.6,
            "the thing is": 0.5,
            "basically": 0.3,
        },
        snap_sentences=True,
        default_order="chronological",
        tail_min=0.8,
        tail_max=3.0,
        tail_settle=0.8,
    ),
    ACTION_CAM: Style(
        name=ACTION_CAM,
        kind=ACTION_CAM,
        # Feature weights (cutpoint.ski): telemetry, VLM tags, audio events,
        # and the voice / visual layers (beauty replaces the VLM scenic tag,
        # wind-gated voice_laughter takes half of the PANNs-event social weight).
        weights={
            "airtime": 0.25,
            "crash": 0.20,
            "vlm": 0.20,
            "speed": 0.15,
            "social": 0.05,
            "voice_laughter": 0.05,
            "carve": 0.05,
            "scenic": 0.0,
            "beauty": 0.05,
            "speech": 0.02,
            "faces": 0.0,
            "stopped": 0.0,
            "short_clip": 0.0,
            "people": 0.0,
            "group": 0.0,
            "voice": 0.0,
        },
        vision_weight=0.2,
        pre_roll=3.0,
        post_roll=3.0,
        min_len=2.0,
        max_len=10.0,
        target_total=150.0,
        merge_gap=1.0,
        keyword_cap=1.5,
        profanity_alone=False,
        keywords=_SKI_KEYWORDS,
        use_scenes=False,
        tail_min=1.0,
        tail_max=3.0,
        must_include_crash=0.6,
    ),
}


def unknown_weights(weights: dict[str, float]) -> list[str]:
    """The names in ``weights`` that are not registered ``grid`` features
    (:mod:`cutpoint.features`; a bonus feature is not a weight), in their order."""
    grid = set(features.names("grid"))
    return [k for k in weights if k not in grid]


def validate(name: str, weights: dict[str, float]) -> None:
    """Check that every weight of the style ``name`` names a registered grid feature
    (:mod:`cutpoint.features`).

    Raises:
        ValueError: Naming the unknown features.
    """
    bad = unknown_weights(weights)
    if bad:
        raise ValueError(
            f"{name}: unknown feature{'s' if len(bad) > 1 else ''} "
            f"{', '.join(map(repr, bad))} in its weights; known: "
            f"{', '.join(features.names('grid'))}"
        )


for _style in STYLES.values():
    validate(_style.name, _style.weights)


@state.register_cache
@functools.cache
def _warn_reserved(path: str) -> None:
    logger.warning("%s ignored: %r is reserved for --style %s", path, AUTO, AUTO)


def learned_styles() -> dict[str, Path]:
    """Learned presets in ``<CUTPOINT_HOME>/styles/*.json`` (name -> file); an ``auto.json``
    is left out with a warning (:data:`AUTO` is reserved)."""
    folder = paths.styles_dir()
    if not folder.is_dir():
        return {}
    found = {p.stem: p for p in sorted(folder.glob("*.json"))}
    if AUTO in found:
        _warn_reserved(str(found.pop(AUTO)))
    return found


def style_names() -> list[str]:
    """Built-in and learned preset names (for ``--style`` choices)."""
    return sorted(STYLES) + [n for n in learned_styles() if n not in STYLES]


@state.register_cache
@functools.cache
def _warn_unknown(path: str, bad: tuple[str, ...]) -> None:
    """Warn once per run: one analyze run and the UI load the same preset many times."""
    one = len(bad) == 1
    logger.warning(
        "%s: unknown feature%s %s in its weights; nothing measures %s, but %s still "
        "count%s, so every score of this style is lower",
        path,
        "" if one else "s",
        ", ".join(map(repr, bad)),
        "it" if one else "them",
        "its weight" if one else "their weights",
        "s" if one else "",
    )


def own_weights(data: dict, base: Style, path: Path) -> dict[str, float]:
    """The weights a style file sets itself (``data["weights"]``), as :func:`load_learned`
    layers them over ``base``: an action-cam file's old ``laughter`` read as
    ``voice_laughter`` and the weights of removed features dropped."""
    own = {k: float(v) for k, v in data.get("weights", {}).items()}
    if base.kind == ACTION_CAM:  # an older preset's "laughter" is voice_laughter (SKI_ALIASES)
        own = features.ski_names(own, str(path))
    gone = [k for k in own if k in features.REMOVED]
    if gone:  # a removed feature (the taste head) scores as if the preset never weighed it
        logger.debug("%s: ignoring the weights of removed features %s", path, gone)
        own = {k: w for k, w in own.items() if k not in features.REMOVED}
    return own


def load_learned(path: Path) -> Style:
    """Build a style from a style file in the styles folder.

    A file with ``"schema"`` (:mod:`cutpoint.stylefiles`) is checked strictly: any problem
    makes the style unusable. A file without it is a legacy preset (``{"name", "base",
    "weights", ...}``, ``scripts/refmatch.py``), read leniently as before: an unknown feature
    in its weights warns and is kept.

    Raises:
        ValueError: If the file is unreadable, its base preset is unknown, or a schema file
            fails its check (naming the file and every problem).
    """
    data = cache.load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Cannot read learned style {path} (missing or invalid JSON)")
    if "schema" in data:
        from cutpoint import stylefiles  # noqa: PLC0415 - it imports this module

        problems = stylefiles.check(data, path.stem).problems
        if problems:
            raise ValueError(f"{path} is not a valid style file: {'; '.join(problems)}")
    if msg := renamed_error(data.get("base")):
        raise ValueError(f'{path}: base {msg}; change its "base"')
    base = STYLES.get(data.get("base", ""))
    if base is None:
        raise ValueError(f"{path}: unknown base style {data.get('base')!r}")
    own = own_weights(data, base, path)
    weights = {**base.weights, **own}
    bad = unknown_weights(weights)
    if bad:  # kept, so the score is as before: counted in the divisor, nothing on the grid
        _warn_unknown(str(path), tuple(bad))
    return replace(
        base,
        name=data.get("name", path.stem),
        weights=weights,
        mask_friends=float(data.get("mask_friends", base.mask_friends)),
        must_include_crash=float(data.get("must_include_crash", base.must_include_crash)),
        learned={k: v for k, v in data.items() if k != "weights"},
        file_hash=content_hash(data),
    )


def content_hash(data: object) -> str:
    """The first 12 hex digits of the SHA-256 of ``data`` as canonical JSON: sorted keys,
    separators ``(",", ":")``, ASCII escapes (``json.dumps`` defaults otherwise)."""
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def style_hash(style: Style) -> str:
    """The ``settings.style_hash`` an analysis records for ``style``.

    A user style: :func:`content_hash` of its file's whole JSON object as read (legacy
    presets too), so any edit to the file changes it. A built-in style: :func:`content_hash`
    of ``{"name", "base", "weights", "mask_friends", "must_include_crash"}`` from its
    definition in :data:`STYLES` (``base`` is its own name), so it changes only when Cutpoint
    changes the style's weights or those overrides, not with a run's
    ``--must-include-crash``.
    """
    if style.file_hash is not None:
        return style.file_hash
    s = STYLES.get(style.name, style)
    return content_hash(
        {
            "name": s.name,
            "base": s.base,
            "weights": s.weights,
            "mask_friends": s.mask_friends,
            "must_include_crash": s.must_include_crash,
        }
    )


def get_style(name: str) -> Style:
    """Return the style preset ``name`` (built-in or ``<styles>/<name>.json``).

    Raises:
        ValueError: If the style is unknown, was renamed (:data:`RENAMED`) or is
            :data:`AUTO` (resolved by the caller, see :func:`resolve_auto`).
    """
    if msg := renamed_error(name):
        raise ValueError(msg)
    if name == AUTO:
        raise ValueError(f"{AUTO!r} detects a style from the footage; it is not a style itself")
    if name in STYLES:
        return STYLES[name]
    learned = learned_styles()
    if name in learned:
        return load_learned(learned[name])
    raise ValueError(f"Unknown style {name!r}; choose from {style_names()}")
