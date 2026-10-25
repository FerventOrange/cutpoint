"""Activity profiles: what an incident (crash, wipeout, fall, fail) looks and sounds like.

A profile is a small JSON document (any sport, game or kind of video)::

    {"name": "mtb", "description": "...",
     "detect": [CLIP captions that recognise this kind of footage],
     "incident": {"pos": [incident captions], "neg": [normal-activity captions],
                  "words": {"over the bars": 1.0, "bail*": 0.8}},
     "highlight": {"pos": [...], "neg": [...]}, "low_info": [...],
     "exclude_base": [base entries this profile replaces], "<section>": {...}}

Every profile inherits the shared floor ``presets/activities/base.json`` (generic
incident captions and crash words); ``exclude_base`` drops floor entries the
profile replaces with its own (the ski profile has snowy versions of all of
them). Extra object sections (e.g. a future ``game`` HUD vocabulary) are kept.

:func:`resolve` picks the profile of a project, in this order:

1. ``activity``: a library profile by name (``--activity``);
2. the cached ``<project>/activity.json`` (unless the request changed or
   ``refresh``);
3. a library match: the cached keyframe CLIP embeddings of the visual layer
   scored zero-shot against every profile's ``detect`` captions; the best
   profile wins when its share and its lead over the runner-up are large enough;
4. synthesis: the local vision model describes a few frames (CLIP zero-shot
   over an open activity vocabulary when it cannot), then the local LLM writes
   a new profile from that description and the user's words (``context``),
   with the built-in profiles as examples. It is validated, capped and saved to
   the library (``<CUTPOINT_HOME>/profiles``), so the next similar video
   matches it in step 3;
5. when the LLM is unavailable or its reply is unusable twice (the retry asks
   for less): a library profile named in ``context`` (``"downhill mtb"`` -> mtb),
   else the nearest one, else ``generic``, recorded as ``source: fallback`` so the
   next run tries again; a WARNING says that ``context`` was not applied.

``context`` always goes through the LLM: with a profile from steps 1-3 its
output is added on top of that profile. Built-in profiles live in
``presets/activities``; library files of the same name override them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, jsonout, paths, state
from cutpoint.config import remap_source
from cutpoint.events.common import load_highlights
from cutpoint.events.visual import load_visual, text_embeddings
from cutpoint.ffmpeg_utils import FFmpegError, grab_frame_jpeg
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
MATCH_SHARE = 0.5  # library match: mean zero-shot share of the best profile ...
MATCH_MARGIN = 0.2  # ... and its lead over the runner-up
DESCRIBE_FRAMES = 4

# (prompt, JPEG frames, JSON schema of the reply or None) -> the model's JSON reply, or None.
Ask = Callable[[str, list[bytes], dict[str, Any] | None], dict[str, Any] | None]
# Synthesis attempts: (incident captions, words, highlight captions) asked for; the
# retry asks for less so a reply cut off by the token limit fits.
SYNTH_SIZES = ((10, 20, 4), (6, 10, 3))
# LLM-written crash words: weight caps (a single word or ``*`` prefix matches normal
# talk more often than a phrase) and filler that says nothing about an incident.
# Sections an LLM-written profile may have (extra sections only come from files).
SCHEMA_KEYS = ("name", "description", "detect", "incident", "highlight", "low_info")
GEN_WORD_CAP = 0.6
GEN_PHRASE_CAP = 0.9
FILLER = {"down", "trail", "ride", "go", "run", "the", "a", "on", "in", "up", "it", "off", "to"}

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
weight 0-1: its crash slang and reactions, e.g. "over the bars", "lost my ski"; a trailing \
* matches word prefixes. Never plain words of the activity or of normal talk such as \
"bike", "down", "air", "rocks" or "trail"; 1.0 only when the phrase means a crash; skip \
generic ones like "you okay" or "crashed", which every profile has),
highlight.pos / highlight.neg (at most {h} captions each: highlight-worthy vs dull moments),
low_info (at most 3 captions of frames with nothing to see).
Captions are short visual descriptions (under 12 words). Where the user says what \
matters, follow it. No other text.

Examples:
{examples}"""

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


def validate(raw: Any) -> dict[str, Any]:
    """Normalise a profile (built-in, library file or LLM output): known fields
    typed, deduplicated and capped; extra object sections kept as they are."""
    raw = raw if isinstance(raw, dict) else {}
    inc = raw.get("incident") if isinstance(raw.get("incident"), dict) else {}
    hl = raw.get("highlight") if isinstance(raw.get("highlight"), dict) else {}
    prof = {
        "name": slugify(str(raw.get("name") or ""), 40, "custom"),
        "description": " ".join(str(raw.get("description") or "").split())[:300],
        "detect": _texts(raw.get("detect"), 6),
        "exclude_base": _texts(raw.get("exclude_base"), MAX_PROMPTS + MAX_WORDS),
        "incident": {
            "pos": _texts(inc.get("pos")),
            "neg": _texts(inc.get("neg")),
            "words": _words(inc.get("words")),
        },
        "highlight": {"pos": _texts(hl.get("pos")), "neg": _texts(hl.get("neg"))},
        "low_info": _texts(raw.get("low_info"), 8),
    }
    prof.update({k: v for k, v in raw.items() if k not in prof and isinstance(v, dict)})
    return prof


def merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``top`` on ``base``: caption lists appended (deduplicated), words
    overridden, identity (name, description, detect) from ``top``; entries in
    ``top["exclude_base"]`` are dropped from ``base``."""
    drop = set(top.get("exclude_base") or ())

    def cat(a: list[str], b: list[str]) -> list[str]:
        out = [s for s in a if s not in drop]
        return out + [s for s in b if s not in out]

    out = {**base, **top, "exclude_base": []}
    for sec in ("incident", "highlight"):
        out[sec] = {k: cat(base[sec][k], top[sec][k]) for k in ("pos", "neg")}
    out["incident"]["words"] = {
        k: w for k, w in base["incident"]["words"].items() if k not in drop
    } | top["incident"]["words"]
    out["low_info"] = cat(base["low_info"], top["low_info"])
    return out


def _load(path: Path) -> dict[str, Any] | None:
    key = str(path)
    if key not in _FILES:
        raw = cache.load_json(path)
        _FILES[key] = validate({**raw, "name": path.stem}) if isinstance(raw, dict) else None
    return _FILES[key]


def builtin(name: str) -> dict[str, Any]:
    """A built-in profile (``presets/activities/<name>.json``); ``base`` is the floor."""
    return _load(BUILTIN_DIR / f"{name}.json")  # type: ignore[return-value]


def base() -> dict[str, Any]:
    """The shared floor."""
    return builtin("base")


def full(profile: dict[str, Any]) -> dict[str, Any]:
    """A library profile with the shared floor applied (what detection uses)."""
    return merge(base(), profile)


def library() -> dict[str, dict[str, Any]]:
    """Built-in profiles, then ``<CUTPOINT_HOME>/profiles/*.json`` (same name overrides)."""
    out: dict[str, dict[str, Any]] = {}
    for d in (BUILTIN_DIR, paths.profiles_dir()):
        for f in sorted(d.glob("*.json")) if d.is_dir() else ():
            prof = _load(f) if f.stem != "base" else None
            if prof is not None:
                out[f.stem] = prof
    return out


def save_to_library(profile: dict[str, Any]) -> str:
    """Store a synthesized profile in the library; returns its (unique) name."""
    name = profile["name"]
    if name == "base" or name in library():
        name = f"{name}-{cache.params_hash(profile, 6)}"
    path = paths.profiles_dir() / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, {**profile, "name": name})
    _FILES.pop(str(path), None)
    return name


# --------------------------------------------------------------------------- footage
def _text_emb(prompts: tuple[str, ...]) -> np.ndarray:
    return text_embeddings(prompts)


def keyframes(project: Path, n: int = 240) -> tuple[np.ndarray, list[tuple[Path, float]]]:
    """Up to ``n`` cached keyframe embeddings ``(N, 768)`` spread over the project's
    sources (visual layer), with their ``(source, time)``."""

    embs, refs = [], []
    for src in dict.fromkeys(s["path"] for s in load_highlights(project)["sources"]):
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


def match_scores(emb: np.ndarray, lib: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Zero-shot share of every library profile with ``detect`` captions."""
    names = [k for k, p in lib.items() if p["detect"]]
    if not len(emb) or not names:
        return {}
    shares = _shares(emb, [lib[k]["detect"] for k in names])
    return {k: round(float(s), 3) for k, s in zip(names, shares, strict=True)}


def nearest(scores: dict[str, float]) -> str:
    """Closest library profile when there is no clear match: the best one with at least
    a fifth of the share, else ``generic``."""
    best = max(scores.items(), key=lambda kv: kv[1], default=("generic", 0.0))
    return best[0] if best[1] >= 0.2 else "generic"


def pick(scores: dict[str, float]) -> str | None:
    """Best profile if it is clear (share and margin), else None; ``generic`` is no match."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][0] == "generic":
        return None
    lead = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return ranked[0][0] if ranked[0][1] >= MATCH_SHARE and lead >= MATCH_MARGIN else None


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
    vocab = [str(v) for v in json.loads((BUILTIN_DIR / "base.json").read_text())["vocabulary"]]
    if not len(emb):
        return "", "none"
    shares = _shares(emb, [[f"a first-person video of {v}"] for v in vocab])
    top = np.argsort(-shares)[:3]
    return "footage of " + ", ".join(f"{vocab[i]} ({shares[i]:.2f})" for i in top), "clip"


def _schema(n: int, w: int, h: int) -> dict[str, Any]:
    """JSON schema of a synthesized profile (Ollama structured output)."""

    def arr(k: int) -> dict[str, Any]:
        return {"type": "array", "items": {"type": "string"}, "maxItems": k}

    def obj(props: dict[str, Any]) -> dict[str, Any]:
        return {"type": "object", "properties": props, "required": list(props)}

    words = {"type": "object", "additionalProperties": {"type": "number"}, "maxProperties": w}
    return obj(
        {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "detect": arr(3),
            "incident": obj({"pos": arr(n), "neg": arr(n), "words": words}),
            "highlight": obj({"pos": arr(h), "neg": arr(h)}),
            "low_info": arr(3),
        }
    )


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
    """A new profile from the LLM (validated and capped), or None after one shorter retry."""
    examples = "\n".join(
        json.dumps({k: v for k, v in s.items() if k != "exclude_base"}) for s in seeds
    )
    for n, w, h in SYNTH_SIZES:
        prompt = _SYNTH_PROMPT.format(
            description=description or "unknown",
            context=context or "nothing",
            examples=examples,
            n=n,
            w=w,
            h=h,
        )
        prof = validate(ask(prompt, [], _schema(n, w, h)))
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
    """The library profile named as a word of ``context`` (``"downhill mtb"`` -> mtb).

    None when the context names no profile or several (``"ski after a hike"``).
    """
    words = re.findall(r"[a-z0-9-]+", context.lower())
    named = {w for w in words if w in lib and w != "generic"}
    return named.pop() if len(named) == 1 else None


def default_ask() -> tuple[Ask | None, str | None]:
    """The local Ollama vision model as an :data:`Ask` (``(None, None)`` if unavailable)."""
    rater = VisionRater()
    return (rater.ask_json, rater.model) if rater.available() else (None, None)


# --------------------------------------------------------------------------- resolve
def resolve(
    project: Path,
    *,
    activity: str | None = None,
    context: str | None = None,
    refresh: bool = False,
    ask: Ask | None = None,
    llm: bool = True,
) -> dict[str, Any]:
    """Profile of a project, cached in ``<project>/activity.json`` (see the module doc).

    Args:
        project: Project folder (highlights.json).
        activity: Library profile name (overrides detection).
        context: The user's own words about the video (fed to the LLM).
        refresh: Ignore activity.json.
        ask: Vision / text model (default: local Ollama when ``llm``).
        llm: Allow the local models (False: library and generic only).

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
    lib = library()
    if activity and activity not in lib:
        raise jsonout.ArgumentError(
            f"unknown activity {activity!r}; known: {', '.join(sorted(lib))}"
        )
    context = " ".join(context.split())[:MAX_CONTEXT_CHARS] if context else None
    request = {"activity": activity, "context": context}
    if (
        old is not None
        and old.get("source") != "fallback"
        and (old.get("request") == request or not (activity or context))
    ):
        return old
    rec: dict[str, Any] = {"version": PROFILE_VERSION, "request": request}
    seed, emb, refs = None, np.zeros((0, 768), np.float32), []
    if activity:
        seed, rec["source"] = activity, "flag"
    else:
        emb, refs = keyframes(project)
        rec["scores"] = match_scores(emb, lib)
        seed = pick(rec["scores"])
        rec["source"] = "library" if seed else "fallback"
    profile = full(lib[seed or nearest(rec["scores"])])
    new = None
    why = "the local models are disabled"  # why ``context`` was not applied
    if (seed is None or context) and llm:
        rec["source"] = "fallback"  # until the LLM delivers (retried on the next run)
        model = None
        if ask is None:
            ask, model = default_ask()
            why = "the Ollama model is unavailable (see the line above)"
        if ask is not None:
            why = "the LLM gave no usable profile"
            if seed:
                desc, how = lib[seed]["description"], "profile"
            else:
                desc, how = describe(emb, refs, ask)
            # Examples: the seed, else the two closest library profiles (the model copies
            # from them, so unrelated ones leak their slang).
            near = sorted(rec.get("scores", {}).items(), key=lambda kv: -kv[1])
            near = [k for k, _ in near if k != "generic"] + ["mtb", "ski"]
            examples = [lib[seed]] if seed else [lib[k] for k in dict.fromkeys(near[:2])]
            new = synthesize(desc, context, ask, examples) if desc or context else None
            rec.update(description=desc, described_by=how, model=model)
            if isinstance(rater := getattr(ask, "__self__", None), VisionRater):
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
    if context and new is None:
        named = _named_profile(context, lib) if seed is None else None
        if named:
            profile = full(lib[named])
        used = f"; using the library profile {named!r} named in it" if named else ""
        message = f"activity: --context not applied: {why}{used}"
        logger.warning("%s", message)
        jsonout.warn("context_not_applied", message)
    rec["name"] = profile["name"]
    rec["profile"] = profile
    cache.save_json(path, rec)
    logger.info("activity: %s (%s)", rec["name"], rec["source"])
    return rec
