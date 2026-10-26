"""Style presets controlling how highlight candidates are scored and cut."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from cutpoint import cache, paths

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
    "laugh*": 1.0,
    "dying": 0.6,
    "i'm dead": 0.8,
}


@dataclass(frozen=True)
class Style:
    """Weights and cut rules for one kind of video.

    Attributes:
        name: Preset name.
        weights: Weight per signal (loudness, speech, laughter, scene,
            continuity); the combined score is their weighted sum, normalised
            to 0-100. ``continuity`` is the fraction of time someone is talking.
        vision_weight: The most a vision model rating can raise or lower a
            candidate's score, as a fraction, when vision is requested
            (``--vision``; opt-in for every style): 0.3 means +30% at 10/10,
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
            or ``"ski"`` (telemetry + audio + VLM features, see
            :mod:`cutpoint.ski`; ``weights`` then name ski features).
        use_scenes: Run scene detection by default (pointless on single-take
            action-cam clips, and the 4K proxy decode is the slowest step).
        learned: Fit metadata for presets loaded from the styles folder.
        mask_friends: Ski only: the lift/stop mask. Above 0 it is hard (no
            clips on lifts or stops; friends in view do not lift it, so every
            value above 0, such as the 5-11 of older presets, acts the same);
            0 = soft mask: nothing is forbidden and the ``stopped`` feature
            weight decides.
        must_include_crash: Ski only: crashes in ``crashes.json`` (``cutpoint
            crashes``) with at least this confidence (0-1) are always clipped,
            with their approach, aftermath and reaction, whatever the weights
            say; 0 = off. Learned presets may set ``"must_include_crash"``.
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
    must_include_crash: float = 0.0

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
    "ski": Style(
        name="ski",
        kind="ski",
        # Feature weights (cutpoint.ski): telemetry, VLM tags, audio events,
        # and the voice / visual layers (beauty replaces the VLM scenic tag,
        # wind-gated laughter takes half of the PANNs-event social weight).
        weights={
            "airtime": 0.25,
            "crash": 0.20,
            "vlm": 0.20,
            "speed": 0.15,
            "social": 0.05,
            "laughter": 0.05,
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
            "taste": 0.0,
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


def learned_styles() -> dict[str, Path]:
    """Learned presets in ``<CUTPOINT_HOME>/styles/*.json`` (name -> file)."""
    folder = paths.styles_dir()
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in sorted(folder.glob("*.json"))}


def style_names() -> list[str]:
    """Built-in and learned preset names (for ``--style`` choices)."""
    return sorted(STYLES) + [n for n in learned_styles() if n not in STYLES]


def load_learned(path: Path) -> Style:
    """Build a style from a learned preset JSON (``{"name", "base", "weights", ...}``).

    Raises:
        ValueError: If the file is unreadable or its base preset is unknown.
    """
    data = cache.load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Cannot read learned style {path} (missing or invalid JSON)")
    base = STYLES.get(data.get("base", ""))
    if base is None:
        raise ValueError(f"{path}: unknown base style {data.get('base')!r}")
    weights = {**base.weights, **{k: float(v) for k, v in data.get("weights", {}).items()}}
    if data.get("taste"):  # personal taste head next to the preset (cutpoint refmatch --taste)
        data["taste"] = str((path.parent / data["taste"]).resolve())
    return replace(
        base,
        name=data.get("name", path.stem),
        weights=weights,
        mask_friends=float(data.get("mask_friends", base.mask_friends)),
        must_include_crash=float(data.get("must_include_crash", base.must_include_crash)),
        learned={k: v for k, v in data.items() if k != "weights"},
    )


def get_style(name: str) -> Style:
    """Return the style preset ``name`` (built-in or ``<styles>/<name>.json``).

    Raises:
        ValueError: If the style is unknown.
    """
    if name in STYLES:
        return STYLES[name]
    learned = learned_styles()
    if name in learned:
        return load_learned(learned[name])
    raise ValueError(f"Unknown style {name!r}; choose from {style_names()}")
