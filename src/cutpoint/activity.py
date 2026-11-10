"""Activity profiles: what an incident (crash, wipeout, fall, fail) looks and sounds like,
and the zero-shot CLIP captions the visual and 360 layers judge this footage with.

A profile is a small JSON document (any sport, game or kind of video)::

    {"name": "<slug>", "description": "...",
     "detect": [CLIP captions that recognise this kind of footage],
     "incident": {"pos": [incident captions], "neg": [normal-activity captions],
                  "words": {"<crash phrase>": 1.0, "<prefix>*": 0.8}},
     "highlight": {"pos": [highlight-worthy views], "neg": [dull ones]},
     "group": {"pos": [friends close to the camera], "neg": [nobody near]},
     "person": {"pos": [another person], "neg": [the wearer's own body, gear, shadow]},
     "selfie": {"pos": [a usable selfie view], "neg": [more captions against it]},
     "lift": {"pos": [riding a lift or shuttle], "neg": [the activity itself]},
     "occluder": [things at the lens], "low_info": [frames with nothing to see],
     "scene": [ordinary footage: negatives of occluder and low_info],
     "vl": {"video": "...", "gear": "...", "scenery": "...", "low_info": "..."},
     "vlm": {"recording": "...", "video": "...", "audio": "...", "describe": "...",
             "hype": "...", "edit": "...", "idle": "...", "rate_edit": "...",
             "rate_high": "...", "rate_low": "...", "retry_high": "...",
             "retry_low": "...", "tags": {"<tag>": "..."}},
     "exclude_base": [base entries this profile replaces], "<section>": {...}}

Every profile inherits the shared floor ``presets/activities/base.json`` (generic
incident captions and crash words, neutral captions of every list); ``exclude_base``
drops floor entries the profile replaces with its own. The floor has no incident
negatives: a full profile without any takes those of ``generic`` (:func:`full`), so its
crash captions are never judged against nothing. ``vl`` is the wording of the 360
vision check (:mod:`cutpoint.pano.vlcheck`); ``vlm`` is the wording of the action-cam VLM
window prompt (:mod:`cutpoint.events.vlm_windows`) and of the action-cam vision rating
(:mod:`cutpoint.vision`), its ``tags`` the descriptions of the fixed window tags
(:data:`cutpoint.events.vlm_windows.ACTION_CAM_TAGS`). A profile's ``vl`` and ``vlm``
strings replace the floor's one by one. ``lift``
is empty in the floor: activities without lifts never look for one. Extra object
sections (e.g. a future ``game`` HUD vocabulary) are kept.

Cutpoint ships no activity profiles of its own: only the floor (``base``) and ``generic``
(the floor plus generic negatives). Every other profile is in the user's library
(``<CUTPOINT_HOME>/profiles``): discovered from footage (step 4 below, ``"source":
"discovered"``) or written by the user or their AI agent (``cutpoint profile create``,
``"source": "hand"``).

:class:`Prompts` holds a profile's caption lists as the CLIP consumers use them
(:func:`prompts`); :func:`use_prompts` makes a set the active one for the visual
and 360 layers (:func:`active_prompts`, else :func:`default_prompts`: ``generic``) and
:func:`project_prompts` reads a project's set from its ``activity.json``.

:func:`resolve` picks the profile of a project, in this order:

1. ``activity``: a library profile by name (``--activity``);
2. the cached ``<project>/activity.json`` (unless the request changed or
   ``refresh``);
3. a library match: the cached keyframe CLIP embeddings of the visual layer
   scored zero-shot against every library profile's ``detect`` captions and the open
   activity vocabulary of ``base.json`` (the terms no library profile covers); the best
   profile wins when its share and its lead over the runner-up are large enough;
4. discovery: the local vision model describes a few frames (CLIP zero-shot
   over the open activity vocabulary when it cannot), then the local LLM writes
   a new profile from that description and the user's words (``context``),
   with ``base`` and ``generic`` as examples. It is validated, capped and saved to
   the library (a file with the same content is reused), so the next similar video
   matches it in step 3. Each synthesis is cached per footage
   (``<cache>/activity/<key>.json``, :func:`discovery_key`: the sources, the model, the
   prompts, ``context`` and the examples), so the same footage gets the same profile
   in every project and run until ``refresh``;
5. when the LLM is unavailable or its reply is unusable twice (the retry asks
   for less): a library profile named in ``context``, else the nearest one, else a
   profile of the best open-vocabulary term when CLIP is clear enough (not saved), else
   ``generic``, recorded as ``source: fallback`` so the next run tries again; a WARNING
   says when ``context`` was not applied.

``context`` always goes through the LLM: with a profile from steps 1-3 its
output is added on top of that profile. Library files named ``generic`` override the
built-in one.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import re
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, jsonout, paths, state
from cutpoint.config import remap_source
from cutpoint.errors import FFmpegError, LowMemoryError
from cutpoint.events.common import load_highlights
from cutpoint.events.visual import load_visual, text_embeddings
from cutpoint.events.vlm_windows import ACTION_CAM_TAGS
from cutpoint.ffmpeg_utils import grab_frame_jpeg
from cutpoint.utils import slugify
from cutpoint.vision import VisionRater

logger = logging.getLogger(__name__)

PROFILE_VERSION = 1
ACTIVITY_FILE = "activity.json"
BUILTIN_DIR = Path(__file__).parent / "presets" / "activities"
MAX_PROMPTS = 16  # per caption list
MAX_WORDS = 40
MAX_PROMPT_CHARS = 100
MAX_WORD_CHARS = 40
MAX_CONTEXT_CHARS = 500
MAX_LIST = 8  # occluder / low_info / scene captions
MAX_EXCLUDE = 80  # exclude_base entries (a profile may replace every floor caption)
MAX_VL_CHARS = 80
MAX_VLM_CHARS = 300  # a vlm string longer than this is dropped (the floor's is used)
MATCH_SHARE = 0.5  # library match: mean zero-shot share of the best profile ...
MATCH_MARGIN = 0.2  # ... and its lead over the runner-up
DESCRIBE_FRAMES = 4
# Read timeout of one synthesis request: a CPU-only Ollama can take minutes per reply, and the
# 600 s default of VisionRater.ask_json would stall `profile` and `crashes` that long (#608).
PROFILE_TIMEOUT_S = 120
# Per-footage cache of discovered profiles (:func:`discovery_key`), under the cache folder.
DISCOVERY_DIR = "activity"
DISCOVERY_VERSION = 1

# (prompt, JPEG frames, JSON schema of the reply or None) -> the model's JSON reply, or None.
Ask = Callable[[str, list[bytes], dict[str, Any] | None], dict[str, Any] | None]
# Synthesis attempts: (incident captions, words, highlight captions, vlm wording) asked
# for; the retry asks for less (no vlm wording) so a reply cut off by the token limit fits.
SYNTH_SIZES = ((10, 20, 4, True), (6, 10, 3, False))
# Sections an LLM-written profile may have (extra sections only come from files).
SCHEMA_KEYS = (
    "name",
    "description",
    "detect",
    "incident",
    "highlight",
    "lift",
    "low_info",
    "vlm",
)
# The profiles Cutpoint ships: the shared floor and ``generic``; every other profile is
# in the user's library (discovered or made by the user or their agent).
BUILTIN_NAMES = ("base", "generic")
# ``source`` of a library file: written by discovery (:func:`save_to_library`) or by
# ``cutpoint profile create``.
DISCOVERED = "discovered"
HAND = "hand"
# The no-LLM fallback (step 5): the best open-vocabulary term needs this share.
VOCAB_SHARE = 0.2
VOCAB_KEY = "vocabulary:"  # match_scores keys of the open vocabulary's terms
# Caption sections of a profile: pairs of lists, plain lists, the vision check's wording.
PAIR_SECTIONS = ("incident", "highlight", "group", "person", "selfie", "lift")
LIST_SECTIONS = ("low_info", "occluder", "scene")
VL_KEYS = ("video", "gear", "scenery", "low_info")
# The action-cam VLM window prompt and vision rating wording (``vlm``), besides its ``tags``.
VLM_KEYS = (
    "recording",
    "video",
    "audio",
    "describe",
    "hype",
    "edit",
    "idle",
    # the vision rating's alone (the window prompt has its own ``edit``)
    "rate_edit",
    "rate_high",
    "rate_low",
    "retry_high",  # the shorter retry prompt's lists
    "retry_low",
)
# The prompt set outside a use_prompts block and for a project without activity.json.
DEFAULT_ACTIVITY = "generic"
# ``llm`` of :func:`resolve` when discovery runs only because a local model may answer (analyze,
# edit): the user did not ask for it, so a missing consent, a bad Ollama port and too little
# free memory are INFO lines and the run goes on as without a model
AUTO = "auto"
# LLM-written crash words: weight caps (a single word or ``*`` prefix matches normal
# talk more often than a phrase) and filler that says nothing about an incident.
GEN_WORD_CAP = 0.6
GEN_PHRASE_CAP = 0.9
FILLER = {"down", "ride", "go", "run", "the", "a", "on", "in", "up", "it", "off", "to"}

_DESCRIBE_PROMPT = (
    "These {n} frames come from one video. Describe the footage for a video editor. "
    "Respond with JSON only: "
    '{{"activity": "...", "setting": "...", "camera": "mount / point of view", '
    '"surface": "terrain or surface", "hazards": "typical crashes, falls or failures"}}'
)
_SYNTH_PROMPT = """You write detection profiles for an automatic video editor. A profile tells a \
CLIP image model and a transcript matcher what an incident (a crash, wipeout, fall, fail or \
death: whatever counts as one in this footage) looks and sounds like.

Footage: {description}
What the user said about it: {context}

Write one compact JSON profile with the keys of the examples:
name (short lowercase slug of the activity), description (one sentence),
detect (3 captions that recognise this kind of footage),
incident.pos (at most {n} captions of frames during or right after an incident, as this \
camera sees them),
incident.neg (at most {n} captions of normal footage of the same activity),
incident.words (at most {w} phrases people say right after an incident in this activity -> \
weight 0-1: its own crash slang and reactions, phrases that name how an incident happens \
here; a trailing * matches word prefixes. Never plain words of the activity or of normal \
talk, such as the names of its gear, its surface or its moves; 1.0 only when the phrase \
means a crash; skip generic ones like "you okay" or "crashed", which every profile has),
highlight.pos / highlight.neg (at most {h} captions each: highlight-worthy vs dull moments),
lift.pos / lift.neg (optional, at most 3 captions each: only when this footage has rides \
on a lift, cable car, shuttle or other transport where the activity pauses; pos: riding \
it, neg: the activity itself; otherwise two empty lists),
low_info (at most 3 captions of frames with nothing to see).{vlm}
Captions are short visual descriptions (under 12 words). Where the user says what \
matters, follow it. No other text.

Examples:
{examples}"""
# Added to _SYNTH_PROMPT on the first attempt: the base wording of the VLM prompts, for the
# LLM to flavour. The tag keys are fixed; only their descriptions may change.
_SYNTH_VLM = """
vlm (optional: the wording a vision model gets for this footage). Start from this generic \
action-cam wording and reword it for this activity, keeping each value's role, length and \
form (hype: a 0-10 rubric; tags: one short description per tag, the same tag keys, no new \
ones; leave out any value that needs no change):
{base_vlm}"""

_FILES: dict[str, dict[str, Any] | None] = state.register_container(__name__, "_FILES", {})


# --------------------------------------------------------------------------- schema
def _texts(v: Any, n: int = MAX_PROMPTS) -> list[str]:
    out: list[str] = []
    for s in v if isinstance(v, list) else []:
        s = " ".join(s.split()) if isinstance(s, str) else ""
        if s and len(s) <= MAX_PROMPT_CHARS and s not in out:
            out.append(s)
    return out[:n]


def _words(v: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, w in (v if isinstance(v, dict) else {}).items():
        k = " ".join(str(k).lower().split())
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if k and len(k) <= MAX_WORD_CHARS and np.isfinite(w) and len(out) < MAX_WORDS:
            out[k] = round(min(1.0, max(0.0, w)), 2)
    return out


def _vlm_text(v: Any, cap: int) -> str:
    """A ``vlm`` string on one line, or "" when it is not a string, empty or over ``cap``."""
    v = " ".join(v.split()) if isinstance(v, str) else ""
    return v if len(v) <= cap else ""


def _vlm(raw: Any) -> dict[str, Any]:
    """The valid part of a ``vlm`` section: the :data:`VLM_KEYS` strings and the
    descriptions of the fixed :data:`ACTION_CAM_TAGS`; a missing, invalid or too long
    value is left out, so the floor's text takes its place."""
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {k: v for k in VLM_KEYS if (v := _vlm_text(raw.get(k), MAX_VLM_CHARS))}
    tags = raw.get("tags") if isinstance(raw.get("tags"), dict) else {}
    out["tags"] = {t: v for t in ACTION_CAM_TAGS if (v := _vlm_text(tags.get(t), MAX_PROMPT_CHARS))}
    return out


def validate(raw: Any) -> dict[str, Any]:
    """Normalise a profile (built-in, library file or LLM output): known fields
    typed, deduplicated and capped; extra object sections kept as they are."""
    raw = raw if isinstance(raw, dict) else {}

    def sec(k: str) -> dict[str, Any]:
        return raw[k] if isinstance(raw.get(k), dict) else {}

    prof: dict[str, Any] = {
        "name": slugify(str(raw.get("name") or ""), 40, "custom"),
        "description": " ".join(str(raw.get("description") or "").split())[:300],
        "detect": _texts(raw.get("detect"), 6),
        "exclude_base": _texts(raw.get("exclude_base"), MAX_EXCLUDE),
    }
    for k in PAIR_SECTIONS:
        prof[k] = {"pos": _texts(sec(k).get("pos")), "neg": _texts(sec(k).get("neg"))}
    prof["incident"]["words"] = _words(sec("incident").get("words"))
    for k in LIST_SECTIONS:
        prof[k] = _texts(raw.get(k), MAX_LIST)
    prof["vl"] = {
        k: " ".join(v.split())[:MAX_VL_CHARS]
        for k in VL_KEYS
        if isinstance(v := sec("vl").get(k), str) and v.strip()
    }
    prof["vlm"] = _vlm(sec("vlm"))
    prof.update({k: v for k, v in raw.items() if k not in prof and isinstance(v, dict)})
    return prof


def merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``top`` on ``base``: caption lists appended (deduplicated), words and ``vl`` /
    ``vlm`` strings overridden (``vlm`` tags one by one), identity (name, description,
    detect) from ``top``; entries in ``top["exclude_base"]`` are dropped from ``base``. A section ``top`` lacks (an LLM
    profile has only :data:`SCHEMA_KEYS`) is the base's."""
    drop = set(top.get("exclude_base") or ())

    def cat(a: list[str], b: list[str]) -> list[str]:
        out = [s for s in a if s not in drop]
        return out + [s for s in b if s not in out]

    def pair(p: dict[str, Any], sec: str, k: str) -> list[str]:
        return (p.get(sec) or {}).get(k) or []

    out = {**base, **top, "exclude_base": []}
    for sec in PAIR_SECTIONS:
        out[sec] = {k: cat(pair(base, sec, k), pair(top, sec, k)) for k in ("pos", "neg")}
    out["incident"]["words"] = {
        k: w for k, w in base["incident"]["words"].items() if k not in drop
    } | top["incident"]["words"]
    for sec in LIST_SECTIONS:
        out[sec] = cat(base.get(sec) or [], top.get(sec) or [])
    out["vl"] = {**(base.get("vl") or {}), **(top.get("vl") or {})}
    bv, tv = base.get("vlm") or {}, top.get("vlm") or {}
    out["vlm"] = {**bv, **tv, "tags": {**(bv.get("tags") or {}), **(tv.get("tags") or {})}}
    return out


def _load(path: Path) -> dict[str, Any] | None:
    key = str(path)
    if key not in _FILES:
        raw = cache.load_json(path)
        _FILES[key] = validate({**raw, "name": path.stem}) if isinstance(raw, dict) else None
    return _FILES[key]


def builtin(name: str) -> dict[str, Any]:
    """A built-in profile (:data:`BUILTIN_NAMES`: ``base``, the floor, or ``generic``)."""
    if name not in BUILTIN_NAMES:
        raise KeyError(f"no built-in activity profile {name!r}")
    return _load(BUILTIN_DIR / f"{name}.json")  # type: ignore[return-value]


def base() -> dict[str, Any]:
    """The shared floor."""
    return builtin("base")


def full(profile: dict[str, Any]) -> dict[str, Any]:
    """A library profile with the shared floor applied (what detection uses); incident
    captions without negatives get those of ``generic`` (:func:`_neg_floor`)."""
    return _neg_floor(merge(base(), profile))


def _neg_floor(profile: dict[str, Any]) -> dict[str, Any]:
    """``profile`` with the incident negatives of ``generic`` (a library file named so
    first) when it has none. The floor has none of its own: a zero-shot crash share
    judged against no negatives is 1.0 on every frame."""
    inc = profile.get("incident")
    if not isinstance(inc, dict) or inc.get("neg"):
        return profile
    lib = library()["generic"]["incident"]["neg"]
    neg = list(lib or builtin("generic")["incident"]["neg"])
    return {**profile, "incident": {**inc, "neg": neg}}


def generic_incident() -> dict[str, Any]:
    """The ``incident`` section of ``--activity generic`` (``pos``, ``neg``, ``words``):
    the floor crash detection never scores below (:class:`cutpoint.crash.CrashParams`)."""
    return full(library()["generic"])["incident"]


def user_files() -> dict[str, Path]:
    """The user library's files (``<CUTPOINT_HOME>/profiles/*.json``, name -> file), sorted;
    a file named ``base`` is never a profile."""
    d = paths.profiles_dir()
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    return {f.stem: f for f in files if f.is_file() and f.stem != "base"}


def library() -> dict[str, dict[str, Any]]:
    """``generic``, then the user library's profiles (a file named ``generic`` overrides it)."""
    out: dict[str, dict[str, Any]] = {"generic": builtin("generic")}
    for stem, f in user_files().items():
        prof = _load(f)
        if prof is not None:
            out[stem] = prof
    return out


def _content(profile: dict[str, Any]) -> dict[str, Any]:
    """A profile as :func:`validate` reads it, without its name."""
    return {k: v for k, v in validate(profile).items() if k != "name"}


def save_to_library(profile: dict[str, Any]) -> str:
    """Store a discovered profile in the library (``"source": "discovered"``); returns its
    (unique) name. A library file with the same content (whatever its name) is reused, so
    the same discovery run twice adds no ``<name>-<hash>`` copy."""
    want = _content(profile)
    for stem, f in user_files().items():
        have = _load(f)
        if have is not None and _content(have) == want:
            return stem
    name = profile["name"]
    taken = {k.lower() for k in library()}  # names that differ only in case clash on Windows
    if name_problem(name) or name.lower() in taken:  # a built-in or a Windows device name
        name = f"{name}-{cache.params_hash(profile, 6)}"
    path = paths.profiles_dir() / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, {**profile, "name": name, "source": DISCOVERED})
    _FILES.pop(str(path), None)
    return name


# --------------------------------------------------------------------------- prompts
@dataclass(frozen=True)
class Prompts:
    """The zero-shot caption lists of one activity, as the CLIP consumers use them.

    ``highlight_*``: beauty / highlight-worthy views (:func:`cutpoint.events.visual.frame_scores`,
    the 360 view scores); ``group_*``: friends close to the camera; ``person_*``: a person
    box is somebody else (pos) or the wearer's own body, gear or shadow (neg);
    ``incident_*``: crash prompts (360 ``crash_zs``); ``occluder``, ``low_info`` and
    ``scene`` (with the group and highlight positives: :meth:`qa_negatives`): the 360 QA
    flags; ``selfie_*``: a usable selfie view (against the occluders too); ``lift_*``:
    riding a lift (empty: no lift detection); ``vl``: the 360 vision check's wording;
    ``vlm`` and ``vlm_tags``: the wording of the action-cam VLM window prompt and vision
    rating (:data:`VLM_KEYS`, the tag descriptions), every value the floor's where the
    profile has none.
    """

    name: str
    highlight_pos: tuple[str, ...]
    highlight_neg: tuple[str, ...]
    group_pos: tuple[str, ...]
    group_neg: tuple[str, ...]
    person_pos: tuple[str, ...]
    person_neg: tuple[str, ...]
    incident_pos: tuple[str, ...]
    incident_neg: tuple[str, ...]
    occluder: tuple[str, ...]
    low_info: tuple[str, ...]
    scene: tuple[str, ...]
    selfie_pos: tuple[str, ...]
    selfie_neg: tuple[str, ...]
    lift_pos: tuple[str, ...]
    lift_neg: tuple[str, ...]
    vl: tuple[tuple[str, str], ...]
    vlm: tuple[tuple[str, str], ...] = ()
    vlm_tags: tuple[tuple[str, str], ...] = ()

    @property
    def lift(self) -> bool:
        """Whether this activity has lifts to look for."""
        return bool(self.lift_pos and self.lift_neg)

    def qa_negatives(self) -> tuple[str, ...]:
        """Negative prompts shared by :attr:`occluder` and :attr:`low_info`."""
        return self.group_pos + self.highlight_pos + self.scene

    def key(self) -> str:
        """Short hash of the captions (not the name or the ``vlm`` wording, which the VLM
        and rating caches key themselves): a cache-key part."""
        skip = ("name", "vlm", "vlm_tags")
        return cache.params_hash({k: v for k, v in asdict(self).items() if k not in skip})


def prompts(profile: dict[str, Any]) -> Prompts:
    """The :class:`Prompts` of a full profile (:func:`full`, ``activity.json``'s ``profile``)."""

    def t(sec: str, k: str | None = None) -> tuple[str, ...]:
        v = profile.get(sec) or ({} if k else [])
        return tuple((v.get(k) or []) if k else v)

    # field by field on the floor's wording: a record's own section may lack values or
    # hold invalid ones (written by hand or before profiles had it)
    own, floor = _vlm(profile.get("vlm")), base()["vlm"]
    vlm = {**floor, **own}
    tags = {**floor["tags"], **own["tags"]}
    return Prompts(
        name=str(profile.get("name") or "custom"),
        highlight_pos=t("highlight", "pos"),
        highlight_neg=t("highlight", "neg"),
        group_pos=t("group", "pos"),
        group_neg=t("group", "neg"),
        person_pos=t("person", "pos"),
        person_neg=t("person", "neg"),
        incident_pos=t("incident", "pos"),
        incident_neg=t("incident", "neg"),
        occluder=t("occluder"),
        low_info=t("low_info"),
        scene=t("scene"),
        selfie_pos=t("selfie", "pos"),
        selfie_neg=t("selfie", "neg"),
        lift_pos=t("lift", "pos"),
        lift_neg=t("lift", "neg"),
        vl=tuple((k, str(v)) for k, v in (profile.get("vl") or {}).items() if k in VL_KEYS),
        vlm=tuple((k, vlm[k]) for k in VLM_KEYS if k in vlm),
        vlm_tags=tuple((t, tags[t]) for t in ACTION_CAM_TAGS if t in tags),
    )


@state.register_cache
@functools.cache
def default_prompts() -> Prompts:
    """The prompt set when none is active: the built-in :data:`DEFAULT_ACTIVITY` profile
    (``generic``: the floor and generic negatives)."""
    return prompts(full(builtin(DEFAULT_ACTIVITY)))


_ACTIVE: ContextVar[Prompts | None] = ContextVar("activity_prompts", default=None)


def active_prompts() -> Prompts:
    """The prompt set of the running command (:func:`use_prompts`), else :func:`default_prompts`."""
    return _ACTIVE.get() or default_prompts()


@contextlib.contextmanager
def use_prompts(p: Prompts) -> Iterator[Prompts]:
    """Inside the block (this thread / task), the visual and 360 layers judge with ``p``."""
    token = _ACTIVE.set(p)
    try:
        yield p
    finally:
        _ACTIVE.reset(token)


def record_profile(rec: dict[str, Any]) -> dict[str, Any]:
    """The full profile of an ``activity.json`` record, with the incident negatives of
    ``generic`` when it has none (a record written before :func:`full` added them).

    A record written before profiles had
    every caption section gets them: a library profile chosen as is (``flag``, ``library``,
    ``fallback``) is rebuilt from the library; one the LLM wrote or refined (``context``,
    ``synthesized``) keeps its identity and ``incident`` and takes every other caption
    section from its library profile or the floor (its old ``highlight`` and ``low_info``
    lists predate the per-activity floor); any other keeps its own sections and takes the
    missing ones from there. A record with every caption section but no ``vlm`` wording
    (written before profiles had one) takes that of its library profile, else the floor's."""
    return _neg_floor(_record_profile(rec))


def _record_profile(rec: dict[str, Any]) -> dict[str, Any]:
    """:func:`record_profile` before the negatives floor."""
    prof = rec.get("profile") if isinstance(rec.get("profile"), dict) else {}
    lib = library()
    name = rec.get("name")
    ref = full(lib[name]) if name in lib else base()
    if all(k in prof for k in (*PAIR_SECTIONS, *LIST_SECTIONS, "vl")):
        return prof if "vlm" in prof else {**prof, "vlm": ref["vlm"]}
    source = rec.get("source")
    if source in ("flag", "library", "fallback") and name in lib:
        return ref
    if source in ("context", "synthesized"):
        keep = ("name", "description", "detect", "incident")
        return {**ref, **{k: prof[k] for k in keep if k in prof}}
    return {**ref, **prof}


def project_prompts(project: Path) -> Prompts:
    """The prompt set of a project's ``activity.json`` (written by ``profile``, ``crashes``
    and the crash step of ``analyze``); :func:`default_prompts` without one. Reads the
    file only: never detects or builds a profile."""
    rec = cache.load_json(project / ACTIVITY_FILE)
    if not isinstance(rec, dict) or rec.get("version") != PROFILE_VERSION:
        return default_prompts()
    return prompts(record_profile(rec))


# --------------------------------------------------------------------------- footage
def _text_emb(prompts: tuple[str, ...]) -> np.ndarray:
    return text_embeddings(prompts)


def keyframes(
    project: Path, n: int = 240, warn_unreachable: bool = True
) -> tuple[np.ndarray, list[tuple[Path, float]]]:
    """Up to ``n`` cached keyframe embeddings ``(N, 768)`` spread over the project's
    sources (visual layer), with their ``(source, time)``. An unreachable source is
    skipped with one WARNING line and one ``footage_unreachable`` JSON warning, or at
    DEBUG without ``warn_unreachable`` (the caller already warned)."""

    embs, refs = [], []
    for src in dict.fromkeys(s["path"] for s in load_highlights(project)["sources"]):
        try:
            cache.source_dir(Path(src))  # the visual cache key needs the file's size and mtime
        except OSError as err:
            if not warn_unreachable:
                logger.debug("%s: footage not reachable (%s); no keyframes from it", src, err)
                continue
            logger.warning("%s: footage not reachable (%s); skipped", src, err)
            jsonout.warn(
                "footage_unreachable", f"{src}: footage not reachable ({err}); skipped", [src]
            )
            continue
        vd = load_visual(Path(src))
        if vd is not None and len(vd.t):
            embs.append(vd.mean_emb)
            refs += [(Path(src), float(t)) for t in vd.t]
    if not embs:
        return np.zeros((0, 768), np.float32), []
    idx = np.unique(np.linspace(0, len(refs) - 1, min(n, len(refs))).astype(int))
    return np.concatenate(embs)[idx], [refs[i] for i in idx]


def _shares(emb: np.ndarray, groups: list[list[str]]) -> np.ndarray:
    """Mean zero-shot softmax share (CLIP logit scale 100) of each caption group."""
    prompts = tuple(s for g in groups for s in g)
    owner = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    logits = 100.0 * emb.astype(np.float32) @ _text_emb(prompts).T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return np.array([p[:, owner == i].sum(1).mean() for i in range(len(groups))])


@state.register_cache
@functools.cache
def vocabulary() -> tuple[str, ...]:
    """The open activity vocabulary of ``base.json`` (CLIP describes footage with it)."""
    raw = cache.load_json(BUILTIN_DIR / "base.json") or {}
    return tuple(str(v) for v in raw.get("vocabulary") or ())


def _vocab_caption(term: str) -> str:
    return f"a first-person video of {term}"


def _covered(lib: dict[str, dict[str, Any]], vocab: tuple[str, ...]) -> set[str]:
    """The vocabulary terms the library profiles cover: per profile (not ``generic``), the
    term whose caption is closest to its ``detect`` captions."""
    names = [k for k, p in lib.items() if p["detect"] and k != "generic"]
    if not names or not vocab:
        return set()
    v = _text_emb(tuple(_vocab_caption(t) for t in vocab))
    return {
        vocab[int(np.argmax((_text_emb(tuple(lib[k]["detect"])) @ v.T).mean(0)))] for k in names
    }


def match_scores(emb: np.ndarray, lib: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Zero-shot share of every library profile with ``detect`` captions, against each other
    and against the open vocabulary's terms no profile covers (keys ``vocabulary:<term>``),
    so the only profile in a library does not win footage of another activity."""
    names = [k for k, p in lib.items() if p["detect"]]
    if not len(emb) or not names:
        return {}
    vocab = vocabulary()
    covered = _covered(lib, vocab)
    terms = [t for t in vocab if t not in covered]
    groups = [lib[k]["detect"] for k in names] + [[_vocab_caption(t)] for t in terms]
    shares = _shares(emb, groups)
    keys = names + [VOCAB_KEY + t for t in terms]
    return {k: round(float(s), 3) for k, s in zip(keys, shares, strict=True)}


def _profiles_only(scores: dict[str, float]) -> list[tuple[str, float]]:
    return [kv for kv in scores.items() if not kv[0].startswith(VOCAB_KEY)]


def nearest(scores: dict[str, float]) -> str:
    """Closest library profile when there is no clear match: the best one with at least
    a fifth of the share, else ``generic``; ``generic`` too when an open-vocabulary term is
    nearer (the vocabulary fallback of :func:`resolve` then names it)."""
    best = max(_profiles_only(scores), key=lambda kv: kv[1], default=("generic", 0.0))
    term = max((s for k, s in scores.items() if k.startswith(VOCAB_KEY)), default=0.0)
    return best[0] if best[1] >= 0.2 and best[1] >= term else "generic"


def pick(scores: dict[str, float]) -> str | None:
    """Best profile if it is clear (share and margin), else None; ``generic`` and an
    open-vocabulary term are no match."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][0] == "generic" or ranked[0][0].startswith(VOCAB_KEY):
        return None
    lead = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return ranked[0][0] if ranked[0][1] >= MATCH_SHARE and lead >= MATCH_MARGIN else None


def vocab_shares(emb: np.ndarray) -> list[tuple[str, float]]:
    """The open vocabulary's terms by zero-shot share over the keyframes, best first."""
    vocab = vocabulary()
    if not len(emb) or not vocab:
        return []
    shares = _shares(emb, [[_vocab_caption(v)] for v in vocab])
    return [(vocab[i], float(shares[i])) for i in np.argsort(-shares)]


def describe(emb: np.ndarray, refs: list[tuple[Path, float]], ask: Ask | None) -> tuple[str, str]:
    """``(description, how)``: the vision model on a few frames (``vl``), else CLIP
    zero-shot over the open activity vocabulary of ``base.json`` (``clip``)."""
    if ask is not None and refs:
        pick_i = np.linspace(0, len(refs) - 1, DESCRIBE_FRAMES + 2)[1:-1].astype(int)
        try:
            imgs = [grab_frame_jpeg(remap_source(refs[i][0]), refs[i][1], 512) for i in pick_i]
        except (OSError, FFmpegError) as err:
            logger.info("activity: frame grab failed (%s); describing with CLIP", err)
            imgs = []
        reply = ask(_DESCRIBE_PROMPT.format(n=len(imgs)), imgs, None) if imgs else None
        if reply:
            text = "; ".join(f"{k}: {reply[k]}" for k in reply if isinstance(reply[k], str))
            if text:
                return text[:400], "vl"
    top = vocab_shares(emb)[:3]
    if not top:
        return "", "none"
    return "footage of " + ", ".join(f"{v} ({s:.2f})" for v, s in top), "clip"


def vocab_profile(emb: np.ndarray) -> dict[str, Any] | None:
    """The no-LLM fallback: a profile of the best open-vocabulary term (its name, a
    description and a ``detect`` caption; every caption list the floor's), or None when no
    term has :data:`VOCAB_SHARE`. Never saved to the library."""
    top = vocab_shares(emb)[:1]
    if not top or top[0][1] < VOCAB_SHARE:
        return None
    term = top[0][0]
    return validate(
        {
            "name": term,
            "description": f"footage of {term} (CLIP, {top[0][1]:.2f})",
            "detect": [_vocab_caption(term)],
        }
    )


def _schema(n: int, w: int, h: int, vlm: bool = False) -> dict[str, Any]:
    """JSON schema of a synthesized profile (Ollama structured output), with the optional
    ``lift`` captions; ``vlm`` adds the optional ``vlm`` wording (its tag keys fixed)."""

    def arr(k: int) -> dict[str, Any]:
        return {"type": "array", "items": {"type": "string"}, "maxItems": k}

    def obj(props: dict[str, Any]) -> dict[str, Any]:
        return {"type": "object", "properties": props, "required": list(props)}

    words = {"type": "object", "additionalProperties": {"type": "number"}, "maxProperties": w}
    schema = obj(
        {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "detect": arr(3),
            "incident": obj({"pos": arr(n), "neg": arr(n), "words": words}),
            "highlight": obj({"pos": arr(h), "neg": arr(h)}),
            "low_info": arr(3),
        }
    )
    # optional: not in "required", so a reply without it is still a profile
    schema["properties"]["lift"] = obj({"pos": arr(3), "neg": arr(3)})
    if vlm:
        text = {"type": "string"}
        tags = {"type": "object", "properties": dict.fromkeys(ACTION_CAM_TAGS, text)}
        schema["properties"]["vlm"] = {
            "type": "object",
            "properties": {**dict.fromkeys(VLM_KEYS, text), "tags": tags},
        }
    return schema


def _cap_words(words: dict[str, float], neg: list[str]) -> dict[str, float]:
    """Generated words capped (single word / prefix 0.6, phrase 0.9); dropped: weight
    0, filler-only phrases, prefixes shorter than 3 letters and single words of the
    normal-activity captions."""
    normal = {t for c in neg for t in c.lower().split()}
    out = {}
    for k, w in words.items():
        toks = k.rstrip("*").split()
        if (
            w <= 0
            or all(t in FILLER for t in toks)
            or (k.endswith("*") and len(k.rstrip("*")) < 3)
            or (len(toks) == 1 and toks[0] in normal)
        ):
            continue
        out[k] = min(w, GEN_WORD_CAP if len(toks) == 1 or "*" in k else GEN_PHRASE_CAP)
    return out


def synthesize(
    description: str, context: str | None, ask: Ask, seeds: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """A new profile from the LLM (validated and capped), or None after one shorter retry.

    ``seeds``: the example profiles, shown with the :data:`SCHEMA_KEYS` the LLM writes
    (without ``vlm``, whose base text the first attempt shows itself). Its ``vlm`` wording
    keeps the valid strings the LLM wrote; merged on the floor (:func:`full`,
    :func:`merge`), every other one is the base text."""
    examples = "\n".join(
        json.dumps({k: s[k] for k in SCHEMA_KEYS if k != "vlm" and k in s}) for s in seeds
    )
    base_vlm = json.dumps(base()["vlm"])
    for n, w, h, vlm in SYNTH_SIZES:
        prompt = _SYNTH_PROMPT.format(
            description=description or "unknown",
            context=context or "nothing",
            examples=examples,
            n=n,
            w=w,
            h=h,
            vlm=_SYNTH_VLM.format(base_vlm=base_vlm) if vlm else "",
        )
        prof = validate(ask(prompt, [], _schema(n, w, h, vlm)))
        if prof["incident"]["pos"]:
            break
        logger.info("activity: no usable profile from the LLM (%d captions asked)", n)
    else:
        return None
    prof = {k: prof[k] for k in SCHEMA_KEYS} | {"exclude_base": []}
    prof["incident"]["words"] = _cap_words(prof["incident"]["words"], prof["incident"]["neg"])
    if not prof["detect"] and prof["description"]:
        prof["detect"] = [prof["description"][:MAX_PROMPT_CHARS]]
    return prof


def _named_profile(context: str, lib: dict[str, dict[str, Any]]) -> str | None:
    """The library profile named as a word of ``context`` (``"race on my-trails"`` ->
    the user's ``my-trails`` profile).

    None when the context names no profile or several.
    """
    words = re.findall(r"[a-z0-9-]+", context.lower())
    named = {w for w in words if w in lib and w != "generic"}
    return named.pop() if len(named) == 1 else None


def default_ask(
    ollama_url: str | None = None, ollama_model: str | None = None, quiet: bool = False
) -> tuple[Ask | None, str | None]:
    """The local Ollama vision model as an :data:`Ask` (``(None, None)`` if unavailable, or
    if that Ollama is on another machine without the user's Vision consent: nothing is sent,
    and :meth:`VisionRater.warn_consent` warns once).

    ``ollama_url``, ``ollama_model``: ``--ollama-url`` / ``--ollama-model``, which win over
    ``config.json`` and the environment (#816). ``(None, None)`` too, with a WARNING, when the
    Ollama URL has a bad port. ``quiet`` (discovery the user did not ask for, :data:`AUTO`):
    the missing consent and the bad port are INFO lines, with no ``--json`` warning."""
    log = logger.info if quiet else logger.warning
    try:
        rater = VisionRater(ollama_url, ollama_model, local_only=True)
    except ValueError as err:  # a bad port in OLLAMA_HOST / ollama_url
        log("Activity detection: Ollama not usable (%s)", err)
        return None, None
    if rater.consent_needed:  # nothing is asked of it, not even /api/tags
        rater.warn_consent(quiet=quiet)
        return None, None
    if not rater.available():
        return None, None
    return functools.partial(rater.ask_json, timeout=PROFILE_TIMEOUT_S), rater.model


# --------------------------------------------------------------------------- discovery cache
def _footage_id(project: Path) -> list[str]:
    """The project's reachable sources as their cache folder names (path, size and
    modification time, :func:`cutpoint.cache.source_dir`), sorted; [] without any."""
    try:
        srcs = load_highlights(project)["sources"]
    except (OSError, ValueError):
        return []
    out = set()
    for s in srcs:
        try:
            out.add(cache.source_dir(Path(s["path"])).name)
        except (OSError, KeyError, TypeError):
            continue  # unreachable: keyframes() warned
    return sorted(out)


def _model_id(ask: Ask, model: str | None) -> dict[str, str | None]:
    """The model behind ``ask``: its name and, from a :class:`VisionRater` that checked it,
    its digest (a model updated under the same name writes the profile again)."""
    rater = getattr(getattr(ask, "func", ask), "__self__", None)
    digest = rater.digest if isinstance(rater, VisionRater) else None
    return {"model": model, "digest": digest}


def discovery_key(
    project: Path,
    model: dict[str, str | None],
    context: str | None,
    examples: list[dict[str, Any]],
) -> str | None:
    """Key of a discovery in the per-footage cache: the project's sources
    (:func:`_footage_id`), the model (:func:`_model_id`), the prompts and attempt sizes
    of the description and the synthesis, the normalised ``context`` and the example
    profiles shown to the LLM. None when no source is reachable (nothing to key it by)."""
    footage = _footage_id(project)
    if not footage:
        return None
    prompts = {
        "describe": _DESCRIBE_PROMPT,
        "frames": DESCRIBE_FRAMES,
        "synth": _SYNTH_PROMPT,
        "vlm": _SYNTH_VLM,
        "base_vlm": base()["vlm"],
        "sizes": SYNTH_SIZES,
        "caps": (GEN_WORD_CAP, GEN_PHRASE_CAP, sorted(FILLER)),
    }
    shown = [{k: s[k] for k in SCHEMA_KEYS if k != "vlm" and k in s} for s in examples]
    return cache.params_hash(
        {
            "v": DISCOVERY_VERSION,
            "footage": footage,
            "model": model,
            "prompts": prompts,
            "context": context,
            "examples": shown,
        },
        16,
    )


def _discovery_path(key: str) -> Path:
    return cache.cache_root() / DISCOVERY_DIR / f"{key}.json"


def _load_discovery(key: str) -> dict[str, Any] | None:
    """A cached discovery (``description``, ``described_by``, ``model``, ``profile``: the
    synthesized profile), or None when there is none or it is not usable."""
    hit = cache.load_json(_discovery_path(key))
    if not isinstance(hit, dict) or hit.get("version") != DISCOVERY_VERSION:
        return None
    prof = hit.get("profile")
    if not isinstance(prof, dict) or not all(k in prof for k in SCHEMA_KEYS):
        return None
    inc = prof["incident"]
    if not isinstance(inc, dict) or not isinstance(inc.get("words"), dict):
        return None
    if not inc.get("pos") or not all(isinstance(inc.get(k, []), list) for k in ("pos", "neg")):
        return None
    return hit


def _save_discovery(key: str, entry: dict[str, Any]) -> None:
    path = _discovery_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, {"version": DISCOVERY_VERSION, **entry})


# --------------------------------------------------------------------------- resolve
def resolve(
    project: Path,
    *,
    activity: str | None = None,
    context: str | None = None,
    refresh: bool = False,
    ask: Ask | None = None,
    llm: bool | str = True,
    warn_unreachable: bool = True,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
) -> dict[str, Any]:
    """Profile of a project, cached in ``<project>/activity.json`` (see the module doc).

    Args:
        project: Project folder (highlights.json).
        activity: Library profile name (overrides detection).
        context: The user's own words about the video (fed to the LLM).
        refresh: Ignore activity.json and the discovery cached for this footage (a new
            synthesis replaces it).
        ask: Vision / text model (default: local Ollama when ``llm``).
        llm: Allow the local models (False: library and generic only; :data:`AUTO`: allowed
            but not asked for, so a missing consent, a bad port or too little free memory
            is an INFO line and the profile is found as without a model, unless
            ``context`` or ``ollama_url`` asks for the model).
        warn_unreachable: Warn about each unreachable source when the footage is read
            (:func:`keyframes`); False when the caller already did.
        ollama_url, ollama_model: ``--ollama-url`` / ``--ollama-model`` for the default
            ``ask`` (None: from config.json and the environment).

    Returns:
        The activity.json record; ``record["profile"]`` is the full profile.
    """
    path = project / ACTIVITY_FILE
    old = None if refresh else cache.load_json(path)
    if not isinstance(old, dict) or old.get("version") != PROFILE_VERSION:
        old = None
    elif old.get("source") == "fallback" and not (activity or context):
        # A failed synthesis is retried with the request that asked for it.
        req = old.get("request") or {}
        activity, context = req.get("activity"), req.get("context")
    context = " ".join(context.split())[:MAX_CONTEXT_CHARS] if context else None
    context = context or None  # blank words are no context
    request = {"activity": activity, "context": context}
    if (
        old is not None
        and old.get("source") != "fallback"
        and (old.get("request") == request or not (activity or context))
    ):
        return old  # it holds the full profile, also of one no longer in the library
    lib = library()
    if activity and activity not in lib:
        if old is None or activity != (old.get("request") or {}).get("activity"):
            raise jsonout.ArgumentError(
                f"unknown activity {activity!r}; known: {', '.join(sorted(lib))}"
            )
        # asked for by an earlier run, the profile is gone (a removed built-in, a deleted file)
        logger.info(
            "activity: profile %r no longer exists; detecting it from the footage", activity
        )
        activity = None
        request = {"activity": None, "context": context}
    rec: dict[str, Any] = {"version": PROFILE_VERSION, "request": request}
    seed, emb, refs = None, np.zeros((0, 768), np.float32), []
    if activity:
        seed, rec["source"] = activity, "flag"
    else:
        emb, refs = keyframes(project, warn_unreachable=warn_unreachable)
        rec["scores"] = match_scores(emb, lib)
        seed = pick(rec["scores"])
        rec["source"] = "library" if seed else "fallback"
    profile = full(lib[seed or nearest(rec["scores"])])
    new = None
    why = "the local models are disabled"  # why ``context`` was not applied
    quiet = llm == AUTO and not (context or ollama_url)  # nobody asked for the model
    if (seed is None or context) and llm:
        rec["source"] = "fallback"  # until the LLM delivers (retried on the next run)
        model = None
        if ask is None:
            # the flags only when given (a stand-in for default_ask takes none)
            flags = {"ollama_url": ollama_url, "ollama_model": ollama_model, "quiet": quiet}
            ask, model = default_ask(**{k: v for k, v in flags.items() if v})
            why = "the Ollama model is unavailable (see the line above)"
        if ask is not None:
            why = "the LLM gave no usable profile"
            try:
                # Examples: the seed, else the floor and generic (no activity's slang to copy).
                floor = [builtin(k) for k in BUILTIN_NAMES]
                if seed and context and not activity and not refresh:
                    # The seed may be this footage's own earlier discovery, saved to the library
                    # (the first project had no match): reuse that, the same profile as then.
                    first = discovery_key(project, _model_id(ask, model), context, floor)
                    if first is not None and _load_discovery(first) is not None:
                        seed = None
                examples = [lib[seed]] if seed else floor
                key = discovery_key(project, _model_id(ask, model), context, examples)
                hit = None if refresh or key is None else _load_discovery(key)
                if hit is not None:  # this footage's profile from an earlier run
                    desc, how, new = hit.get("description"), hit.get("described_by"), hit["profile"]
                    rec.update(description=desc, described_by=how, model=hit.get("model"))
                    logger.info("activity: discovered profile of this footage reused (%s)", key)
                else:
                    if seed:
                        desc, how = lib[seed]["description"], "profile"
                    else:
                        desc, how = describe(emb, refs, ask)
                    new = synthesize(desc, context, ask, examples) if desc or context else None
                    rec.update(description=desc, described_by=how, model=model)
                    if new is not None and key is not None:  # never a fallback
                        entry = {"description": desc, "described_by": how, "model": model}
                        _save_discovery(key, {**entry, "profile": new})
            except LowMemoryError as err:
                if not quiet:
                    raise
                logger.info("activity: %s; detecting it without the vision model", err)
            finally:
                target = getattr(ask, "func", ask)  # default_ask() wraps it in a partial
                if isinstance(rater := getattr(target, "__self__", None), VisionRater):
                    rater.unload()  # done with it: VRAM for the CLIP steps, other models stay
            if new is not None:  # LLM words add phrases, never re-weight known ones
                known = (profile if seed else base())["incident"]["words"]
                new["incident"]["words"] = {
                    k: w for k, w in new["incident"]["words"].items() if k not in known
                }
            if new is not None and seed:
                profile = merge(
                    profile, {**new, **{k: profile[k] for k in ("name", "description", "detect")}}
                )
                rec["source"] = "context"
            elif new is not None:
                seed = save_to_library(new)
                profile = full({**new, "name": seed})
                rec["source"] = "synthesized"
    named = _named_profile(context, lib) if context and new is None and seed is None else None
    if named:
        profile = full(lib[named])
    if context and new is None:
        used = f"; using the library profile {named!r} named in it" if named else ""
        message = f"activity: --context not applied: {why}{used}"
        logger.warning("%s", message)
        jsonout.warn("context_not_applied", message)
    if seed is None and new is None and not named and profile["name"] == "generic":
        # no model, nothing near in the library: the best open-vocabulary term, if clear
        guess = vocab_profile(emb)
        if guess is not None:
            profile = full(guess)
            rec.setdefault("description", guess["description"])
            rec.setdefault("described_by", "clip")
    rec["name"] = profile["name"]
    rec["profile"] = profile
    cache.save_json(path, rec)
    logger.info("activity: %s (%s)", rec["name"], rec["source"])
    return rec


# --------------------------------------------------------------------------- library commands
# ``cutpoint profile list | show | create | validate | delete`` (the user library).
def name_problem(name: object) -> str | None:
    """Why ``name`` cannot name a library profile, or None: a safe file name (as a style
    name) that is not ``base`` or ``generic`` (compared ignoring case)."""
    from cutpoint.stylefiles import _RESERVED_STEMS, NAME_RE  # noqa: PLC0415

    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        return (
            f"{name!r} is not a usable profile name: use up to 64 letters, digits, '-' and "
            "'_', starting with a letter or digit"
        )
    if name.lower() in BUILTIN_NAMES:
        return f"{name!r} is a built-in profile name; choose another name"
    if name.lower() in _RESERVED_STEMS:
        return f"{name!r} cannot be a file name on Windows; choose another name"
    return None


def _file_source(raw: Any) -> str | None:
    """A library file's ``source`` (:data:`DISCOVERED`, :data:`HAND`), None when it has none
    (written before files had one)."""
    src = raw.get("source") if isinstance(raw, dict) else None
    return src if src in (DISCOVERED, HAND) else None


def _check_list(raw: Any, key: str, cap: int, problems: list[str], warnings: list[str]) -> None:
    """Check one caption list ``raw`` (named ``key``) as :func:`validate` reads it."""
    if raw is None:
        return
    if not isinstance(raw, list):
        problems.append(f"{key} must be a list of captions")
        return
    kept = _texts(raw, cap)
    if len(kept) < len(raw):
        warnings.append(
            f"{key}: {len(raw) - len(kept)} of {len(raw)} entries ignored (not text, empty, "
            f"longer than {MAX_PROMPT_CHARS} characters, repeated or over the {cap} kept)"
        )


def _check_words(inc: dict[str, Any], problems: list[str], warnings: list[str]) -> None:
    words = inc.get("words")
    if words is not None and not isinstance(words, dict):
        problems.append('incident.words must be an object of "phrase": weight (0-1)')
        return
    if not words:
        return
    kept = _words(words)
    if len(kept) < len(words):
        warnings.append(
            f"incident.words: {len(words) - len(kept)} of {len(words)} entries ignored "
            f"(weight not a number, phrase longer than {MAX_WORD_CHARS} characters or over "
            f"the {MAX_WORDS} kept)"
        )
    out = [k for k, w in words.items() if isinstance(w, int | float) and not 0 <= w <= 1]
    if out:
        warnings.append(f"incident.words: weights outside 0-1 clamped: {', '.join(out)}")


def _check_texts(raw: dict[str, Any], problems: list[str], warnings: list[str]) -> None:
    """The ``vl`` and ``vlm`` wording."""
    for sec, keys, cap in (("vl", VL_KEYS, MAX_VL_CHARS), ("vlm", VLM_KEYS, MAX_VLM_CHARS)):
        val = raw.get(sec)
        if val is None:
            continue
        if not isinstance(val, dict):
            problems.append(f"{sec} must be an object of texts")
            continue
        unknown = sorted(set(val) - set(keys) - ({"tags"} if sec == "vlm" else set()))
        if unknown:
            warnings.append(f"{sec}: unknown keys ignored: {', '.join(unknown)}")
        bad = [k for k in keys if k in val and not (isinstance(val[k], str) and val[k].strip())]
        if bad:
            warnings.append(f"{sec}: not text, ignored: {', '.join(bad)}")
        long = [
            k for k in keys if isinstance(val.get(k), str) and len(" ".join(val[k].split())) > cap
        ]
        if long:
            what = "cut to" if sec == "vl" else "ignored, longer than"
            warnings.append(f"{sec}: {what} {cap} characters: {', '.join(long)}")
        tags = val.get("tags") if sec == "vlm" else None
        if tags is not None and not isinstance(tags, dict):
            problems.append("vlm.tags must be an object of tag descriptions")
        elif tags and (unknown := sorted(set(tags) - set(ACTION_CAM_TAGS))):
            warnings.append(
                f"vlm.tags: unknown tags ignored: {', '.join(unknown)} (the tags are "
                f"{', '.join(ACTION_CAM_TAGS)})"
            )


def check_profile(raw: Any) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` of a profile document, in plain words: a problem makes it
    unusable; a warning names what :func:`validate` ignores, cuts or caps when it loads it."""
    if not isinstance(raw, dict):
        return ["the file is not a JSON object"], []
    problems: list[str] = []
    warnings: list[str] = []
    _check_list(raw.get("detect"), "detect", 6, problems, warnings)
    _check_list(raw.get("exclude_base"), "exclude_base", MAX_EXCLUDE, problems, warnings)
    for sec in PAIR_SECTIONS:
        val = raw.get(sec)
        if val is None:
            continue
        if not isinstance(val, dict):
            problems.append(f'{sec} must be an object with "pos" and "neg" lists')
            continue
        for k in ("pos", "neg"):
            _check_list(val.get(k), f"{sec}.{k}", MAX_PROMPTS, problems, warnings)
        unknown = sorted(set(val) - {"pos", "neg", *(["words"] if sec == "incident" else [])})
        if unknown:
            warnings.append(f"{sec}: unknown keys ignored: {', '.join(unknown)}")
    for sec in LIST_SECTIONS:
        _check_list(raw.get(sec), sec, MAX_LIST, problems, warnings)
    _check_words(
        raw["incident"] if isinstance(raw.get("incident"), dict) else {}, problems, warnings
    )
    _check_texts(raw, problems, warnings)
    known = {"name", "description", "source", "detect", "exclude_base", "vl", "vlm"}
    known |= {*PAIR_SECTIONS, *LIST_SECTIONS}
    dropped = sorted(k for k, v in raw.items() if k not in known and not isinstance(v, dict))
    if dropped:
        warnings.append(f"unknown keys ignored: {', '.join(dropped)}")
    if not problems:
        prof = validate(raw)
        captions = [prof["detect"], *(prof[s][k] for s in PAIR_SECTIONS for k in ("pos", "neg"))]
        if not any(captions) and not prof["incident"]["words"]:
            problems.append("the profile has no captions and no incident words")
    return problems, warnings


def _entry(name: str, path: Path | None) -> dict[str, Any]:
    """One :func:`list_profiles` entry."""
    if path is None:
        return {
            "name": name,
            "type": "built-in",
            "source": "built-in",
            "floor": name == "base",
            "valid": True,
            "problems": [],
            "description": builtin(name)["description"],
            "path": None,
        }
    raw = cache.load_json(path)
    problems, _ = check_profile(raw)
    return {
        "name": name,
        "type": "user",
        "source": _file_source(raw),
        "floor": False,
        "valid": not problems,
        "problems": problems,
        "description": validate(raw)["description"],
        "path": path,
    }


def list_profiles() -> list[dict[str, Any]]:
    """The built-in ``base`` (the floor every profile inherits, not a choice for
    ``--activity``) and ``generic``, then the user library's files (sorted).

    Each entry: ``name``, ``type`` (``built-in`` or ``user``), ``source`` (``built-in``,
    :data:`DISCOVERED`, :data:`HAND`, or None for a file without one), ``floor``, ``valid``,
    ``problems``, ``description`` and ``path`` (None for a built-in). A broken file is
    listed as not valid, never raised.
    """
    return [_entry(n, None) for n in BUILTIN_NAMES] + [
        _entry(stem, path) for stem, path in user_files().items()
    ]


def _find(name: str) -> Path:
    """The library file of the profile ``name``.

    Raises:
        FileNotFoundError: If the library has no such file.
    """
    path = user_files().get(name)
    if path is None:
        raise FileNotFoundError(
            f"no profile named {name!r} in your library ({paths.profiles_dir()}); "
            "`cutpoint profile list` shows the profiles"
        )
    return path


def show_profile(name: str) -> dict[str, Any]:
    """The profile ``name``: a file of the user library, else ``base`` or ``generic``.

    Returns:
        ``name``, ``type``, ``source``, ``floor``, ``path``, ``description`` and ``profile``
        (its own content as detection reads it, before the floor is added).

    Raises:
        FileNotFoundError: For an unknown name.
        jsonout.ArgumentError: For a library file that cannot be used.
    """
    path = user_files().get(name)
    if path is None and name not in BUILTIN_NAMES:
        path = _find(name)
    entry = _entry(name, path)
    if not entry["valid"]:
        raise jsonout.ArgumentError(f"{path} cannot be used: {'; '.join(entry['problems'])}")
    prof = builtin(name) if path is None else _load(path)
    keys = ("name", "type", "source", "floor", "path", "description")
    return {k: entry[k] for k in keys} | {"profile": prof}


def create_profile(name: str, data: Any, *, force: bool = False) -> tuple[Path, list[str]]:
    """Save ``data`` as the library profile ``<profiles>/<name>.json`` (``"source":
    "hand"``), as :func:`validate` reads it; ``name`` replaces the document's own.

    Returns:
        The file written and the warnings of :func:`check_profile`.

    Raises:
        jsonout.ArgumentError: For an unusable name or content (naming every problem).
        FileExistsError: If the profile exists and ``force`` is not given.
    """
    if msg := name_problem(name):
        raise jsonout.ArgumentError(msg)
    problems, warnings = check_profile(data)
    if problems:
        raise jsonout.ArgumentError(f"profile {name!r} not saved: {'; '.join(problems)}")
    files = user_files()
    clash = next((s for s in files if s.lower() == name.lower() and s != name), None)
    if clash is not None:
        raise jsonout.ArgumentError(
            f"a profile named {clash!r} exists; names that differ only in case clash on "
            "Windows and macOS"
        )
    path = paths.profiles_dir() / f"{name}.json"
    if name in files and not force:
        raise FileExistsError(
            f"profile {name!r} already exists ({path}); add --force to replace it"
        )
    prof = validate({**data, "name": name})
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, {"name": name, "source": HAND} | prof)
    _FILES.pop(str(path), None)
    return path, warnings


def validate_profile(target: str) -> dict[str, Any]:
    """Check the profile ``target``: a profile name, or a file (a path, or a name ending in
    ``.json``).

    Returns:
        ``name``, ``path`` (None for a built-in), ``type`` (``built-in``, ``user`` for a
        library file, ``file`` for another), ``valid``, ``problems``, ``warnings`` (``code``,
        ``message``) and ``notes``. The caller fails the command when ``valid`` is false.

    Raises:
        FileNotFoundError: For an unknown profile name or a missing file.
    """
    as_file = target.endswith(".json") or "/" in target or "\\" in target
    if not as_file and target in BUILTIN_NAMES and target not in user_files():
        return {
            "name": target,
            "path": None,
            "type": "built-in",
            "valid": True,
            "problems": [],
            "warnings": [],
            "notes": [f"{target!r} is a built-in profile"],
        }
    if as_file or Path(target).is_file():
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"{target}: file not found")
    else:
        path = _find(target)
    problems, warnings = check_profile(cache.load_json(path))
    in_library = path.parent.resolve() == paths.profiles_dir().resolve()
    notes = []
    if not in_library and not problems:
        notes.append(f"`cutpoint profile create NAME --from {path}` saves it to your library")
    return {
        "name": path.stem,
        "path": path,
        "type": "user" if in_library else "file",
        "valid": not problems,
        "problems": problems,
        "warnings": [{"code": "profile_entry_ignored", "message": w} for w in warnings],
        "notes": notes,
    }


def delete_profile(name: str) -> Path:
    """Delete the library profile ``name`` (its file in the profiles folder).

    Raises:
        jsonout.ArgumentError: For ``base`` or ``generic`` without a library file.
        FileNotFoundError: If there is no such file.
    """
    if name in BUILTIN_NAMES and name not in user_files():
        raise jsonout.ArgumentError(f"{name!r} is a built-in profile and cannot be deleted")
    path = _find(name)
    path.unlink()
    _FILES.pop(str(path), None)
    return path
