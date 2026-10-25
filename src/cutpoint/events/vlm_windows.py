"""Layer 2: describe candidate windows with Qwen2.5-Omni (frames + audio) via llama-server.

For each window (default 15 s) one OpenAI-compatible chat request carries:
    * 12-16 frames sampled evenly at 720p, each labelled with its timestamp,
    * the window's audio as 16 kHz mono WAV (``input_audio``),
    * the Whisper transcript excerpt and layer-1 detector hints (as text),
and asks for strict JSON (enforced with a JSON-schema grammar).

Audio needs Qwen's official system prompt: without it, Qwen2.5-Omni often
claims it cannot hear audio or hallucinates (verified with llama.cpp b11748).

Window selection merges three seed sources: highlights.json candidates,
layer-1 audio events (laughter/scream/cheer) and OCR events (kills, banners).
Windows extend ``pre_s`` before and ``post_s`` after a seed, overlapping
windows are merged up to ``max_len_s``, and the best ``max_windows`` are kept.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint import cache
from cutpoint.config import source_path
from cutpoint.events.common import (
    Source,
    audio_slice,
    layer_cache_path,
    transcript_excerpt,
    wav_bytes,
)
from cutpoint.events.eventtypes import EventType, first_match
from cutpoint.events.llama_server import LlamaServer, LlamaServerError
from cutpoint.events.regions import HudPreset, is_installed_path, load_preset
from cutpoint.ffmpeg_utils import FFmpegError, grab_images
from cutpoint.utils import clock

logger = logging.getLogger(__name__)

QWEN_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
PROMPT_VERSION = 5

# Property order matters: llama.cpp's grammar emits keys in this order, so the
# model describes the events before it commits to the scores.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "number"},
                    "what": {"type": "string"},
                    "funny": {"type": "integer"},
                    "hype": {"type": "integer"},
                },
                "required": ["t", "what", "funny", "hype"],
            },
        },
        "summary": {"type": "string"},
        "funny_score": {"type": "integer"},
        "highlight_score": {"type": "integer"},
        "label": {"type": "string"},
    },
    "required": ["events", "summary", "funny_score", "highlight_score", "label"],
}

# Ski (``style_kind == "ski"``): per-window tag strengths for the ski scorer.
SKI_PROMPT_VERSION = 1
SKI_TAGS = (
    "jump_air",
    "landing",
    "crash_fall",
    "powder_spray",
    "fast_groomer",
    "trees",
    "scenic_vista",
    "friends_faces",
    "chairlift",
    "standing_talking",
    "lodge",
)
RESPONSE_SCHEMA_SKI: dict[str, Any] = {
    "type": "object",
    "properties": {
        **RESPONSE_SCHEMA["properties"],
        "tags": {
            "type": "object",
            "properties": {t: {"type": "integer"} for t in SKI_TAGS},
            "required": list(SKI_TAGS),
        },
    },
    "required": [*RESPONSE_SCHEMA["required"], "tags"],
}
# Move "tags" before the scores so the model looks before it rates.
RESPONSE_SCHEMA_SKI["properties"] = {
    k: RESPONSE_SCHEMA_SKI["properties"][k]
    for k in ("events", "summary", "tags", "funny_score", "highlight_score", "label")
}
HYPE_SKI = (
    "  hype: 0-1 standing around, riding a lift, indoors; 3-4 ordinary cruising; "
    "6-7 fast riding, powder spray, tight trees, a small jump; 8-10 a big jump, "
    "a crash or fall, a spectacular line, a near miss.\n"
)
SKI_TAG_HELP = (
    "Tags (integers 0-10, how clearly each is present in this window; 0 = absent): "
    "jump_air (the rider is airborne), landing, crash_fall (someone falls or crashes), "
    "powder_spray (snow spraying from turns in deep snow), fast_groomer (fast riding "
    "on a smooth groomed run), trees (riding between trees), scenic_vista (a wide "
    "mountain view worth showing), friends_faces (other people clearly visible, "
    "faces, interacting with the camera), chairlift (on a chairlift or gondola), "
    "standing_talking (stopped, standing around or talking), lodge (indoors, lodge, "
    "restaurant).\n"
)

HYPE_GENERIC = (
    "  hype: 0-1 driving/waiting, menus, hangar, loading, vehicle or respawn "
    "select, scoreboard, killcam, replay; 3-4 a single ordinary kill; 6-7 a "
    "multi-kill, a big explosion, a close call, a match win; 8-10 several "
    "kills in quick succession, a spectacular kill or a spectacular death.\n"
)


@dataclass(frozen=True)
class GamePrompt:
    """Game-specific parts of the VLM prompt (from the HUD preset).

    Attributes:
        name: Game name ("My Game", "video game").
        view: Camera view ("first-person"; empty: not mentioned).
        hype: The hype rubric line.
        custom: A project preset (not an installed user or bundled one) gives a
            prompt (name, view, hype) that differs from the reference prompt of its
            game key (the installed preset's, else the derived default), so the VLM
            cache key includes the prompt text. Event type seeds do not count: they
            only change window selection.
    """

    name: str = "video game"
    view: str = ""
    hype: str = HYPE_GENERIC
    custom: bool = False

    def key(self) -> str:
        """Short hash of the prompt fields (VLM cache key ``gp``)."""
        text = json.dumps([self.name, self.view, self.hype])
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _prompt_of(name: str, preset: HudPreset | None) -> GamePrompt:
    """Prompt fields of ``preset`` (``name``: the derived game name); a ``hype`` line
    is normalised to two leading spaces and one trailing newline."""
    if preset is None:
        return GamePrompt(name)
    hype = preset.hype.strip()
    return GamePrompt(
        name=preset.title or name,
        view=preset.view,
        hype=f"  {hype}\n" if hype else HYPE_GENERIC,
    )


def game_prompt(game: str | None, preset: HudPreset | None) -> GamePrompt:
    """The prompt fields for ``game``: the preset's ``title``/``view``/``hype``, else
    the game key as a name ("some-game" -> "Some Game") and the generic rubric.

    ``custom`` is set for a project preset whose prompt differs from the
    reference: the installed (user, else bundled) preset of the same game key,
    else the derived default.
    Presets without prompt fields (e.g. written by discovery) keep their VLM cache keys.
    """
    name = (game or "").replace("-", " ").title() or "video game"
    prompt = _prompt_of(name, preset)
    if preset is None or is_installed_path(preset.path):
        return prompt
    ref = _prompt_of(name, load_preset(game))
    if prompt == ref:
        return prompt
    return GamePrompt(prompt.name, prompt.view, prompt.hype, custom=True)


@dataclass
class VlmParams:
    """Parameters for window selection and VLM requests."""

    window_s: float = 15.0
    pre_s: float = 3.0
    post_s: float = 12.0
    cand_post_s: float = 10.0  # extend highlights.json candidates past their end
    max_len_s: float = 36.0  # merged windows longer than this are not merged further
    max_windows: int = 60
    frames: int = 14  # for windows up to 18 s; longer ones get up to 16
    forced: tuple[tuple[str, float, float], ...] = ()  # (name glob, start, end)
    frame_height: int = 720
    image_max_tokens: int = 640
    ctx: int = 16384
    audio: bool = True
    transcript: bool = True
    hints: bool = True
    max_tokens: int = 700
    ski: bool = False  # ski prompt + tags (set by the pipeline for ski projects)
    max_fill_windows: int = 0  # ski: extra windows over stopped / lift footage

    def key(self, prompt: GamePrompt | None = None) -> dict[str, Any]:
        """Parameters that change a window's result (selection params excluded).

        A ``prompt`` from a custom project preset adds its hash as ``gp``.
        """
        return {
            "frames": self.frames,
            "h": self.frame_height,
            "imt": self.image_max_tokens,
            "audio": self.audio,
            "transcript": self.transcript,
            "hints": self.hints,
            "prompt": PROMPT_VERSION,
            "model": "qwen2.5-omni-7b-q4km",
            **({"ski": SKI_PROMPT_VERSION} if self.ski else {}),
            **({"gp": prompt.key()} if prompt and prompt.custom else {}),
        }


@dataclass
class Window:
    """A time window of one source to describe."""

    source: str
    start: float
    end: float
    priority: float
    reasons: list[str]

    @property
    def key(self) -> str:
        """Cache key within a source."""
        return f"{self.start:.1f}-{self.end:.1f}"


# --------------------------------------------------------------------------- selection

SEED_WEIGHTS = {"Laughter": 1.0, "Scream": 0.7, "Cheering": 0.6}
TELEMETRY_SEEDS = {"crash": 2.0, "airtime": 1.0, "coverage": 0.25}
# Stopped / lift footage (ski): described only where no other window exists, so
# the mask exception (friends on the lift, banter at the top) can see it.
GAP_FILL_TYPES = ("stopped/lift",)


def _seed_priority(ev: dict[str, Any], types: tuple[EventType, ...] = ()) -> float | None:
    """Priority of a layer-1 event as a window seed (None = not a seed).

    The ``seed`` of the first matching preset event type in ``types`` comes first.
    """
    t = first_match(types, ev)
    if t and t.seed is not None:
        return t.seed
    if ev.get("kind") == "audio":
        w = SEED_WEIGHTS.get(ev["label"])
        if w is None or not ev.get("seed", True):
            return None
        return w * (0.5 + min(1.0, ev["confidence"] / 0.3))
    if ev.get("kind") == "ocr":
        typ, text = ev.get("type"), (ev.get("text") or "").upper()
        if typ == "banner":
            if "REVERSE FRIENDLY FIRE" in text or text == "ACE" or "CLUTCH" in text:
                return 2.0
            if any(
                k in text
                for k in ("WON ROUND", "ROUND WON", "VICTORY", "DEFEAT", "LOST ROUND", "ROUND LOST")
            ):
                return 1.3
            if text in ("MATCH POINT", "OVERTIME", "ELIMINATED"):
                return None
            return 0.6
        if typ == "kill":
            if ev.get("teamkill"):
                return 2.0
            return 1.2 if ev.get("player") else 0.5
        if typ == "vs":
            m = re.match(r"(\d) VS (\d)", text)
            if m and (m.group(1) == "1" or m.group(2) == "1"):
                return 0.8  # clutch situation
            return None
        if typ == "log":
            return 0.9 if ev.get("player") else None
        if typ == "multikill":
            return 2.0
    if ev.get("kind") == "telemetry" and ev.get("type") == "coverage_masked":
        return 0.1
    if ev.get("kind") == "telemetry":
        base = TELEMETRY_SEEDS.get(ev.get("type", ""))
        if base is None:
            return None
        return base + (ev.get("strength") or 0) if ev.get("type") == "airtime" else base
    return None


def cached_windows(sources: list[Source], params: VlmParams) -> list[Window]:
    """Windows already described with these parameters (from the per-source cache)."""
    out = []
    for s in sources:
        store = cache.load_json(layer_cache_path(s.cdir, "vlm", params.key())) or {}
        for rec in (store.get("windows") or {}).values():
            out.append(Window(str(s.path), rec["start"], rec["end"], 0.0, ["cached"]))
    return out


def add_incremental(
    prev: list[Window], new: list[Window], min_overlap: float = 0.5
) -> list[Window]:
    """``prev`` plus the windows of ``new`` that ``prev`` does not already cover.

    Ski windows are seeded partly by highlights.json candidates, which change
    once VLM tags feed the scorer; reusing described windows keeps reruns
    cheap and only fills footage nobody has looked at yet.
    """
    keep = list(prev)
    for w in new:
        cov = sum(
            max(0.0, min(w.end, p.end) - max(w.start, p.start))
            for p in prev
            if p.source == w.source
        )
        if cov < min_overlap * (w.end - w.start):
            keep.append(w)
    return keep


def select_windows(
    sources: list[Source],
    candidates: list[dict[str, Any]],
    layer1: dict[str, list[dict[str, Any]]],
    params: VlmParams,
    types: dict[str, tuple[EventType, ...]] | None = None,
    quiet: bool = False,
) -> list[Window]:
    """Build, merge and cap the windows to send to the VLM.

    Args:
        sources: Project sources (for durations).
        candidates: highlights.json candidates.
        layer1: ``{source_path: [layer-1 events]}`` (audio + OCR).
        params: Selection parameters.
        types: ``{source_path: event types}`` (custom seed priorities).
        quiet: Log the window counts at debug level (VLM layer off: only cached
            descriptions are reused).
    """
    dur = {str(s.path): s.duration for s in sources}
    raw: list[Window] = []
    for c in candidates:
        src = c["source"]
        if src not in dur:
            continue
        # Candidates come from dialogue/loudness peaks; the payoff (a kill
        # after the banter) often follows, so extend past the candidate end.
        a = c["start"] - 1.0
        b = max(c["end"] + params.cand_post_s, a + params.window_s)
        b = min(b, a + params.max_len_s)
        raw.append(
            Window(
                src,
                a,
                b,
                0.4 + c.get("score", 0) / 100,
                [f"cand {c.get('id')} {c.get('label', '')}".strip()],
            )
        )
    for src, events in layer1.items():
        src_types = (types or {}).get(src, ())
        for ev in events:
            pr = _seed_priority(ev, src_types)
            if pr is None:
                continue
            t = ev.get("peak_t", ev["t"])
            if ev.get("kind") == "telemetry" and ev.get("type") in ("coverage", "coverage_masked"):
                a, b = ev["t"], ev["end"]
            else:
                a, b = t - params.pre_s, t + params.post_s
            raw.append(
                Window(
                    src,
                    a,
                    b,
                    pr,
                    [f"{ev['kind']}:{ev.get('label') or ev.get('text') or ev.get('type')}"],
                )
            )
    for pattern, fa, fb in params.forced:
        for src in dur:
            if fnmatch.fnmatch(source_path(src).name.lower(), pattern.lower()):
                raw.append(Window(src, fa, fb, 99.0, [f"forced {pattern}@{fa:g}-{fb:g}"]))
    for w in raw:
        d = dur.get(w.source, w.end)
        w.start, w.end = max(0.0, w.start), min(d, w.end)
    raw = [w for w in raw if w.end - w.start >= 3.0]
    fill = [w for w in raw if any(r.endswith(GAP_FILL_TYPES) for r in w.reasons)]
    raw = [w for w in raw if w not in fill]
    raw.sort(key=lambda w: (w.source, w.start))
    merged: list[Window] = []
    for w in raw:
        prev = merged[-1] if merged else None
        forced = any(r.startswith("forced") for r in (prev.reasons if prev else []) + w.reasons)
        if (
            prev
            and prev.source == w.source
            and w.start <= prev.end
            and (
                max(prev.end, w.end) - prev.start <= params.max_len_s
                or (forced and w.end <= prev.end)
            )
        ):
            prev.end = max(prev.end, w.end)
            prev.priority = max(prev.priority, w.priority) + 0.15 * min(prev.priority, w.priority)
            prev.reasons += w.reasons
        else:
            merged.append(Window(w.source, w.start, w.end, w.priority, list(w.reasons)))
    merged.sort(key=lambda w: -w.priority)
    chosen = merged[: params.max_windows]
    extra = 0
    for w in fill:  # gap fill: never merged, so existing windows keep their cache keys
        if extra >= params.max_fill_windows:
            break
        if not any(c.source == w.source and w.start < c.end and c.start < w.end for c in chosen):
            chosen.append(w)
            extra += 1
    order = {str(s.path): i for i, s in enumerate(sources)}
    chosen.sort(key=lambda w: (order.get(w.source, 0), w.start))
    for w in chosen:
        w.start, w.end = round(w.start, 1), round(w.end, 1)
    logger.log(
        logging.DEBUG if quiet else logging.INFO,
        "VLM windows: %d raw seeds -> %d merged -> %d kept (+%d gap-fill)",
        len(raw),
        len(merged),
        len(chosen) - extra,
        extra,
    )
    return chosen


# --------------------------------------------------------------------------- media


def frames_for(dur: float, base: int) -> int:
    """Frames for a window: ``base`` up to 18 s, then ~1 per 2.25 s, at most 16."""
    if dur <= 18.0:
        return base
    return max(base, min(16, round(dur / 2.25)))


def grab_frames(
    path: Path, start: float, dur: float, n: int, height: int, hwaccel: bool = True
) -> list[bytes]:
    """Decode ``n`` evenly spaced JPEG frames from ``[start, start + dur]``.

    Frame ``k`` is at ``start + k * dur / n``. Only this window is read.
    """
    try:
        return grab_images(
            path,
            start,
            count=n,
            duration=dur,
            fps=f"{n}/{dur:.3f}",
            height=height,
            quality=4,
            hwaccel="cuda" if hwaccel else None,
        )
    except FFmpegError as err:
        logger.debug("frame grab failed: %s", err)
        return []


def _hints(events: list[dict[str, Any]], start: float, end: float) -> str:
    lines = []
    for ev in events:
        t = ev.get("peak_t", ev["t"])
        if not start - 1 <= t <= end:
            continue
        if ev["kind"] == "audio" and ev["label"] in SEED_WEIGHTS:
            lines.append(f"{t - start:.1f}s audio detector: {ev['label']} ({ev['confidence']:.2f})")
        elif ev["kind"] == "telemetry" and ev.get("type") in ("airtime", "crash"):
            lines.append(f"{t - start:.1f}s camera telemetry: {ev['label']}")
        elif ev["kind"] == "ocr" and ev["type"] in (
            "kill",
            "banner",
            "vs",
            "log",
            "bigtext",
            "crit",
            "fire",
            "multikill",
        ):
            extra = " (TEAMKILL)" if ev.get("teamkill") else ""
            # feed kills (generic verb feeds and built-in R6 rows): say whose kill it is
            if ev.get("verb") or ev["type"] == "kill":
                extra += (
                    " (the PLAYER's kill)"
                    if ev.get("player_kill")
                    else " (the PLAYER died)"
                    if ev.get("player_death")
                    else ""
                )
            lines.append(f"{t - start:.1f}s on-screen {ev['type']}: {ev['text']}{extra}")
    return "\n".join(lines[:15])


def build_messages(
    src: Source,
    win: Window,
    frames: list[bytes],
    params: VlmParams,
    game: GamePrompt | str | None,
    transcript_lines: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """OpenAI-style messages for one window.

    ``game`` is the window's :class:`GamePrompt`, or a game key whose preset is
    looked up (bundled and user presets).
    """
    if params.ski:
        return build_messages_ski(src, win, frames, params, transcript_lines, events)
    if not isinstance(game, GamePrompt):
        game = game_prompt(game, load_preset(game))
    dur = win.end - win.start
    view = f"{game.view} " if game.view else ""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"This is a {dur:.0f}-second window of a {view}{game.name} recording, "
                f"starting at {clock(win.start)} in {src.name}. The single audio track mixes "
                "game sound, the player's microphone and friends on voice chat. "
                f"{len(frames)} frames follow, each labelled with seconds from the window start."
            ),
        }
    ]
    content += _media_content(src, win, frames, params, transcript_lines, events)
    content.append(
        {
            "type": "text",
            "text": (
                "Describe what actually happens in this window, in time order: what the "
                "player does, kills and deaths (who kills whom, with what), teamkills, round "
                "results, and how the friends react (laughing, shouting, banter, jokes). "
                "Do not list transcript lines one by one: summarise a conversation as ONE "
                "event and spend the other events on actions and reactions. Connect the "
                "talk to the action: if a conversation is followed by a kill, a death, a "
                "round win or laughter, say so explicitly ('after the argument, the player "
                "gets a kill and the team laughs'). Look at the frames for weapons, kills "
                "and deaths, and listen to the audio for laughter and tone. "
                "Use on-screen HUD text only as context; do not transcribe it.\n"
                "Scoring rubric (integers 0-10):\n"
                + game.hype
                + "  funny: 0-1 no joke; 3-4 light banter; 6-7 a clear joke or people "
                "laughing; 8-10 everyone laughing hard, an absurd mistake, a teamkill "
                "followed by laughter.\n"
                "  highlight_score: how much this window deserves a place in a highlight "
                "montage (max of the best moment's hype and funny, plus 1 if both).\n"
                "Event 't' is seconds from the window start. Reply with JSON only: "
                '{"events": [{"t": number, "what": str, "funny": int, '
                '"hype": int}], "summary": str, "funny_score": int, '
                '"highlight_score": int, "label": "2-4 word slug"}'
            ),
        }
    )
    return [{"role": "system", "content": QWEN_SYSTEM}, {"role": "user", "content": content}]


def _media_content(
    src: Source,
    win: Window,
    frames: list[bytes],
    params: VlmParams,
    transcript_lines: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Frames, audio, transcript and hints of a window (shared by the prompts)."""
    dur = win.end - win.start
    content: list[dict[str, Any]] = []
    step = dur / max(1, len(frames))
    for i, jpg in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i + 1} @ {i * step:.1f}s:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()},
            }
        )
    if params.audio:
        samples = audio_slice(src, win.start, win.end)
        if len(samples):
            content.append({"type": "text", "text": "Audio of the whole window:"})
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(wav_bytes(samples)).decode(),
                        "format": "wav",
                    },
                }
            )
    if params.transcript:
        ex = transcript_excerpt(transcript_lines, src.name, win.start, win.end)
        content.append(
            {
                "type": "text",
                "text": "Speech transcript (automatic, may contain "
                "errors; seconds from window start):\n" + (ex or "(none)"),
            }
        )
    if params.hints:
        h = _hints(events, win.start, win.end)
        if h:
            content.append(
                {"type": "text", "text": "Detector hints (automatic, may be wrong):\n" + h}
            )
    return content


def build_messages_ski(
    src: Source,
    win: Window,
    frames: list[bytes],
    params: VlmParams,
    transcript_lines: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ski/snowboard variant of :func:`build_messages` (adds the ``tags`` rubric)."""
    dur = win.end - win.start
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"This is a {dur:.0f}-second window of a first-person GoPro skiing / snowboarding "
                f"recording (helmet or chest mount), starting at {clock(win.start)} in "
                f"{src.name}. The audio is the camera microphone: wind noise at speed, the "
                f"rider and friends nearby. {len(frames)} frames follow, each labelled with "
                "seconds from the window start."
            ),
        }
    ]
    content += _media_content(src, win, frames, params, transcript_lines, events)
    content.append(
        {
            "type": "text",
            "text": (
                "Describe what actually happens in this window, in time order: the terrain "
                "(groomed run, trees, powder, park), jumps (take-off, air, landing), falls, "
                "friends on camera and what they do, lift rides, standing around or talking, "
                "indoor scenes and views. Camera telemetry hints (accelerometer) are reliable "
                "for airtime and impacts. Listen for laughter, cheering and screams.\n"
                + SKI_TAG_HELP
                + "Scoring rubric (integers 0-10):\n"
                + HYPE_SKI
                + "  funny: 0-1 nothing funny; 3-4 light banter; 6-7 a clear joke or people "
                "laughing; 8-10 everyone laughing hard, a ridiculous fall.\n"
                "  highlight_score: how much this window deserves a place in a ski highlight "
                "edit (lifts, standing around and lodge scenes score low unless friends do "
                "something funny).\n"
                "Event 't' is seconds from the window start. Reply with JSON only: "
                '{"events": [{"t": number, "what": str, "funny": int, '
                '"hype": int}], "summary": str, "tags": {<tag>: int, ...}, '
                '"funny_score": int, "highlight_score": int, "label": "2-4 word slug"}'
            ),
        }
    )
    return [{"role": "system", "content": QWEN_SYSTEM}, {"role": "user", "content": content}]


def parse_reply(text: str) -> dict[str, Any] | None:
    """Parse and sanitise the model's JSON reply (None if unusable)."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

    def score(v: Any) -> int:
        try:
            return max(0, min(10, round(float(v))))
        except (TypeError, ValueError):
            return 0

    events = []
    for ev in data.get("events") or []:
        if isinstance(ev, dict) and ev.get("what"):
            try:
                t = float(ev.get("t", 0))
            except (TypeError, ValueError):
                t = 0.0
            events.append(
                {
                    "t": t,
                    "what": str(ev["what"])[:200],
                    "funny": score(ev.get("funny")),
                    "hype": score(ev.get("hype")),
                }
            )
    label = re.sub(r"[^a-z0-9]+", "-", str(data.get("label", "")).lower()).strip("-")
    out = {
        "summary": str(data.get("summary", ""))[:600],
        "events": events,
        "highlight_score": score(data.get("highlight_score")),
        "funny_score": score(data.get("funny_score")),
        "label": label[:40],
    }
    if isinstance(data.get("tags"), dict):
        out["tags"] = {t: score(data["tags"].get(t)) for t in SKI_TAGS}
    return out


# --------------------------------------------------------------------------- driver


def describe_windows(
    server: LlamaServer | None,
    sources: dict[str, Source],
    windows: list[Window],
    params: VlmParams,
    prompts: dict[str, GamePrompt],
    transcript_lines: list[dict[str, Any]],
    layer1: dict[str, list[dict[str, Any]]],
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Describe every window (cached per source + window + parameters).

    Args:
        server: Running llama-server, or None to use cached results only.
        sources: ``{source_path: Source}``.
        windows: Windows from :func:`select_windows`.
        params: VLM parameters.
        prompts: ``{source_path: game prompt}`` (missing: a generic video game).
        transcript_lines: Parsed transcript.txt.
        layer1: Layer-1 events per source (for hints).
        force: Ignore cached results.

    Returns:
        (window results, stats).
    """
    results: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "requested": len(windows),
        "ok": 0,
        "failed": 0,
        "cached": 0,
        "seconds": [],
        "prompt_tokens": [],
    }
    caches: dict[str, tuple[Path, dict[str, Any]]] = {}
    for win in windows:
        src = sources[win.source]
        prompt = prompts.get(win.source)
        if win.source not in caches:
            cpath = layer_cache_path(src.cdir, "vlm", params.key(prompt))
            caches[win.source] = (cpath, cache.load_json(cpath) or {"windows": {}})
        cpath, store = caches[win.source]
        hit = store["windows"].get(win.key)
        if hit and not force:
            stats["cached"] += 1
            results.append({**hit, "reasons": win.reasons, "priority": win.priority})
            continue
        if server is None:
            continue
        t0 = time.perf_counter()
        frames = grab_frames(
            src.io_path,
            win.start,
            win.end - win.start,
            frames_for(win.end - win.start, params.frames),
            params.frame_height,
        )
        if not frames:
            logger.warning("%s %s: no frames decoded; skipping", src.name, win.key)
            stats["failed"] += 1
            continue
        msgs = build_messages(
            src,
            win,
            frames,
            params,
            prompt,
            transcript_lines,
            layer1.get(win.source, []),
        )
        parsed, usage, err = None, {}, None
        for attempt in range(2):
            try:
                resp = server.chat(
                    msgs,
                    max_tokens=params.max_tokens,
                    temperature=0.2 if attempt == 0 else 0.0,
                    json_schema=(RESPONSE_SCHEMA_SKI if params.ski else RESPONSE_SCHEMA),
                )
                usage = resp.get("usage", {})
                parsed = parse_reply(resp["choices"][0]["message"].get("content", ""))
                if parsed:
                    break
            except (LlamaServerError, KeyError, ValueError) as e:  # retry once
                err = str(e)
                logger.warning("%s %s: VLM request failed: %s", src.name, win.key, e)
                if "audio" in err.lower() and params.audio:
                    raise
        elapsed = round(time.perf_counter() - t0, 1)
        if not parsed:
            stats["failed"] += 1
            logger.warning("%s %s: unusable VLM reply (%s)", src.name, win.key, err)
            continue
        rec = {
            "source": win.source,
            "start": win.start,
            "end": win.end,
            **parsed,
            "frames": len(frames),
            "elapsed_s": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
        }
        store["windows"][win.key] = rec
        cache.save_json(cpath, {**store, "params": params.key(prompt)})
        stats["ok"] += 1
        stats["seconds"].append(elapsed)
        stats["prompt_tokens"].append(usage.get("prompt_tokens"))
        results.append({**rec, "reasons": win.reasons, "priority": win.priority})
        logger.info(
            "VLM %s %s-%s (%.1fs, %s tok): hl %d funny %d '%s' %s",
            src.name,
            clock(win.start),
            clock(win.end),
            elapsed,
            usage.get("prompt_tokens"),
            parsed["highlight_score"],
            parsed["funny_score"],
            parsed["label"],
            parsed["summary"][:100],
        )
    return results, stats


def windows_to_events(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten VLM window results into events (absolute time)."""
    out = []
    for r in results:
        dur = r["end"] - r["start"]
        for ev in r["events"]:
            t = r["start"] + min(max(0.0, ev["t"]), dur)
            out.append(
                {
                    "source": r["source"],
                    "t": round(t, 2),
                    "end": round(t, 2),
                    "kind": "vlm",
                    "type": "vlm",
                    "label": ev["what"],
                    "funny": ev["funny"],
                    "hype": ev["hype"],
                    "confidence": round(max(ev["funny"], ev["hype"]) / 10, 2),
                }
            )
    return out


def check_audio(server: LlamaServer, samples: Any) -> dict[str, Any]:
    """Send one short audio clip; report whether the server accepts audio input.

    Args:
        server: Running llama-server.
        samples: 16 kHz mono float32 samples (ideally a clip with laughter).

    Returns:
        ``{"accepted": bool, "reply" | "error": str}``.
    """
    msgs = [
        {"role": "system", "content": QWEN_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(wav_bytes(samples)).decode(),
                        "format": "wav",
                    },
                },
                {
                    "type": "text",
                    "text": "Is anyone laughing in this audio? Answer yes or "
                    "no, then describe the vocal sounds in one "
                    "sentence.",
                },
            ],
        },
    ]
    try:
        resp = server.chat(msgs, max_tokens=80, temperature=0.0)
        return {"accepted": True, "reply": resp["choices"][0]["message"]["content"]}
    except (LlamaServerError, KeyError) as err:
        return {"accepted": False, "error": str(err)[:300]}
