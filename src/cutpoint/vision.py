"""Optional frame rating with a local Ollama vision model (e.g. qwen3.5:9b)."""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from cutpoint import cache, paths
from cutpoint.backends import ollama_url
from cutpoint.ffmpeg_utils import FFmpegError, grab_frame_jpeg

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "qwen3.5:9b"


def default_model() -> str:
    """Vision model: ``CUTPOINT_VISION_MODEL`` or qwen3.5:9b."""
    return paths.env("VISION_MODEL") or FALLBACK_MODEL


_PROMPT = (
    "These are {n} frames, in order, from one short moment of a {style} video. "
    "Rate how highlight-worthy this moment is for a {style} highlight reel "
    "(0 = boring, 10 = must include). Live action scores high: fights, kills, "
    "clutches, chases, big plays. Non-gameplay scores low (0-2): menus, "
    "operator/character select, post-round or post-match screens, scoreboards, "
    "loading screens, killcams and replays. Respond with JSON only: "
    '{{"score": <0-10 number>, "label": "<2-4 word description>"}}'
)
_PROMPT_SKI = (
    "These are {n} frames, in order, from one short moment of a first-person skiing / "
    "snowboarding video. Rate how highlight-worthy it is for a ski edit (0 = boring, "
    "10 = must include). High: jumps, crashes, powder spray, fast riding, tight trees, "
    "friends doing something funny, a stunning view. Low (0-2): riding a chairlift, "
    "standing around, the lodge, the ground or the sky only. Respond with JSON only: "
    '{{"score": <0-10 number>, "label": "<2-4 word description>"}}'
)
# Retry prompt: shorter, asks for no reasoning. Thinking variants of qwen3-vl
# reason even with think=false; a long reasoning trace used to fill the
# default 4096-token context and the reply came back empty (done_reason=length).
_RETRY_PROMPT = (
    "{n} frames from a {style} video moment. Answer immediately, no reasoning. "
    "Score 0-10 for a highlight reel; menus, scoreboards, loading screens and "
    "replays score 0-2. Output only JSON: "
    '{{"score": <number>, "label": "<2-4 words>"}}'
)
_RETRY_PROMPT_SKI = (
    "{n} frames from a first-person skiing / snowboarding video moment. Answer "
    "immediately, no reasoning. Score 0-10 for a ski edit; jumps, crashes, powder and "
    "fast riding score high, chairlifts, standing around and the lodge score 0-2. "
    'Output only JSON: {{"score": <number>, "label": "<2-4 words>"}}'
)
# (first prompt, retry prompt) per style; styles not listed use the default pair.
_PROMPTS = {"ski": (_PROMPT_SKI, _RETRY_PROMPT_SKI)}
_DEFAULT_PROMPTS = (_PROMPT, _RETRY_PROMPT)
# (num_ctx, num_predict) per attempt. 3 frames at 640 px are ~3300 prompt
# tokens, so num_ctx must leave room for the reasoning plus the JSON.
_ATTEMPTS = ((8192, 3072), (12288, 6144))
_TEMPERATURES = (0.2, 0.4)  # sampling temperature per attempt
FRAME_WIDTH = 640  # px of each frame sent to the model


@dataclass(frozen=True)
class RatingProfile:
    """How a model family is asked for a rating where it differs from the default
    (rate()'s n_frames, each frame a separate image, the reply's score as given)."""

    n_frames: int  # frames per rating, in place of rate()'s n_frames
    cols: int  # the frames go as one contact sheet, this many per row, in reading order
    expected: bool  # score = expected value of the score digit over its top logprobs


# Per model family: a model whose name starts with the key (case-insensitive). With the default
# input (separate frames, the score as given) qwen3.5:9b ranks moments clearly worse than
# qwen3-vl:8b against hand labels and gives most moments the same score; six frames on one 3x2
# sheet of 640 px tiles (about 1500 prompt tokens) plus the expected score over the digit's
# logprobs closes most of the gap at about 1.3 s per rating, against 12 s for qwen3-vl (#475).
_PROFILES = {"qwen3.5": RatingProfile(n_frames=6, cols=3, expected=True)}
_TOP_LOGPROBS = 10  # alternatives Ollama returns per token for RatingProfile.expected
SHEET_QUALITY = 90  # JPEG quality of a contact sheet
# Ratings cached per source (cache.source_dir), one entry per rating input (rating_key),
# stamped with the model's digest (VisionRater._lookup says how the digest is matched).
RATINGS_FILE = "vision-ratings.json"
# Bump when the rating input or the reply parsing changes outside what rating_key hashes
# (the prompts, attempt options and temperatures, frame width, rating profile, model, style
# and frame times).
RATING_VERSION = 1


def rating_profile(model: str) -> RatingProfile | None:
    """The model's :class:`RatingProfile`, or None for the default input."""
    name = model.lower()
    return next((p for key, p in _PROFILES.items() if name.startswith(key)), None)


@dataclass
class VisionResult:
    """Rating returned by the vision model."""

    score: float
    label: str


class VisionRater:
    """Client for Ollama's /api/chat endpoint with images."""

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or ollama_url()).rstrip("/")
        if not self.url.startswith("http"):
            self.url = f"http://{self.url}"
        self.model = model or default_model()
        self.used = False  # True once a request went to the model: unload() frees it
        self.hits = 0  # ratings taken from the cache by rate()
        self.asked = 0  # ratings rate() asked the model for (failed ones included)
        self.missed = 0  # cache misses rate() gave no rating for (cache_only)
        # the model's digest from /api/tags (set by available()); None: not known
        self.digest: str | None = None
        # True: rate() gives only cached ratings and asks the model nothing (Ollama down)
        self.cache_only = False
        # per source: (cache file, its ratings), or None when the source cannot be keyed
        self._ratings: dict[Path, tuple[Path, dict[str, Any]] | None] = {}

    def available(self) -> bool:
        """True if the server answers and has the model pulled; records the model's
        ``digest`` for the rating cache."""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"/api/tags returned {type(data).__name__}, not an object")
            models = data.get("models", [])
        except (requests.RequestException, ValueError) as err:  # ValueError: not JSON
            logger.info("Ollama not reachable at %s (%s); skipping vision", self.url, err)
            return False
        names = {m.get("name", "") for m in models}
        if self.model not in names and f"{self.model}:latest" not in names:
            logger.info(
                "Ollama model %s not pulled (have: %s); skipping vision",
                self.model,
                ", ".join(sorted(names)) or "none",
            )
            return False
        found = (m for m in models if m.get("name") in (self.model, f"{self.model}:latest"))
        digest = next(found, {}).get("digest")
        self.digest = digest if isinstance(digest, str) and digest else None
        return True

    def ask_json(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        schema: dict | None = None,
        num_predict: int = 4096,
    ) -> dict | None:
        """One turn (text, optionally with JPEG frames); the reply's JSON object or None.

        ``schema`` constrains the reply (Ollama structured output; default: any JSON).
        Thinking builds of Qwen3 reason even with ``think: false`` until ``num_predict``
        runs out, so text-only turns to a Qwen model (name starting with ``qwen``) go
        through ``/api/generate`` in raw mode with the ChatML turn and an empty
        ``<think></think>`` block already written; other models use ``/api/chat``.
        """
        options = {"temperature": 0.2, "num_ctx": 8192, "num_predict": num_predict}
        payload: dict = {"model": self.model, "stream": False, "format": schema or "json"}
        raw = not images and self.model.lower().startswith("qwen")
        if not raw:
            msg = {"role": "user", "content": prompt}
            if images:
                msg["images"] = [base64.b64encode(b).decode("ascii") for b in images]
            payload |= {"messages": [msg], "think": False, "options": options}
            endpoint = "chat"
        else:
            payload |= {
                "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "raw": True,
                "options": options,
            }
            endpoint = "generate"
        self.used = True
        try:
            resp = requests.post(f"{self.url}/api/{endpoint}", json=payload, timeout=600)
            resp.raise_for_status()
            body = resp.json()
            content = body["response"] if raw else body["message"]["content"]
            data = _json_object(content)
        except (requests.RequestException, KeyError, ValueError) as err:
            logger.warning("Ollama request failed: %s", err)
            return None
        return data if isinstance(data, dict) else None

    def rate(
        self, src: Path, start: float, end: float, style: str, n_frames: int = 3
    ) -> VisionResult | None:
        """Grab ``n_frames`` across [start, end] and ask the model for a rating.

        Ratings are cached per source (``RATINGS_FILE`` in :func:`cutpoint.cache.source_dir`)
        under :func:`rating_key`, each written as soon as the model gives it, so a rerun asks
        the model only about new frames. A failed grab or request is not cached. With
        ``cache_only`` a miss gives None (counted in ``missed``) and nothing is sent. The
        model's :func:`rating_profile` sets the number of frames in place of ``n_frames``.
        """
        times = _frame_times(start, end, self._n_frames(n_frames))
        store = self._store(src)
        key = rating_key(self.model, style, times)
        if (hit := self._lookup(store, key)) is not None:
            self.hits += 1
            return hit
        if self.cache_only:
            self.missed += 1
            return None
        result = self._rate_frames(src, times, style)
        if result is not None and store is not None:
            entry: dict[str, Any] = {"score": result.score, "label": result.label}
            store[1][key] = entry | ({"digest": self.digest} if self.digest else {})
            try:
                cache.save_json(store[0], store[1])
            except OSError as err:  # a cache must not fail the run
                logger.warning("Could not save the vision rating cache %s: %s", store[0], err)
        return result

    def cached(
        self, src: Path, start: float, end: float, style: str, n_frames: int = 3
    ) -> VisionResult | None:
        """The rating :meth:`rate` would take from the cache, or None; counts nothing."""
        key = rating_key(self.model, style, _frame_times(start, end, self._n_frames(n_frames)))
        return self._lookup(self._store(src), key)

    def _n_frames(self, n_frames: int) -> int:
        """Frames per rating: the model's profile's, else ``n_frames``."""
        profile = rating_profile(self.model)
        return profile.n_frames if profile else n_frames

    def _lookup(self, store: tuple[Path, dict[str, Any]] | None, key: str) -> VisionResult | None:
        """The cached rating under ``key``, or None.

        The key holds the model's tag (its name) and each entry the digest of the model that
        gave it. With the digest known (Ollama answered), only an entry with that digest
        matches, so a model updated under the same tag rates again; an entry without a digest
        (cached before digests were recorded) is a miss too. With no digest to compare
        (``cache_only``: Ollama or the model is not available; or /api/tags listed none), the
        entry for the tag matches whatever its digest: a new rating replaces the entry, so it
        is the latest rating made with that tag.
        """
        if store is None:
            return None
        entry = store[1].get(key)
        if self.cache_only or self.digest is None:
            return _cached(entry)
        if not isinstance(entry, dict) or entry.get("digest") != self.digest:
            return None
        return _cached(entry)

    def _store(self, src: Path) -> tuple[Path, dict[str, Any]] | None:
        """The cached ratings of ``src`` (loaded once; a missing or corrupt file is empty);
        None if the source cannot be keyed (e.g. missing: the frame grab reports it)."""
        if src not in self._ratings:
            try:
                path = cache.source_dir(src) / RATINGS_FILE
            except OSError as err:
                logger.debug("No vision rating cache for %s: %s", src, err)
                self._ratings[src] = None
            else:
                data = cache.load_json(path)
                self._ratings[src] = (path, data if isinstance(data, dict) else {})
        return self._ratings[src]

    def _rate_frames(self, src: Path, times: list[float], style: str) -> VisionResult | None:
        """Grab the frames at ``times`` and ask the model for a rating (no cache)."""
        n_frames = len(times)
        profile = rating_profile(self.model)
        try:
            frames = [grab_frame_jpeg(src, t, width=FRAME_WIDTH) for t in times]
            if profile:
                frames = [contact_sheet(frames, profile.cols)]
        except (FFmpegError, ValueError) as err:  # ValueError: a frame cv2 cannot decode
            logger.warning("Vision frame grab failed: %s", err)
            return None
        images = [base64.b64encode(b).decode("ascii") for b in frames]
        self.asked += 1
        content = ""
        steps = zip(_prompts(style, profile), _ATTEMPTS, _TEMPERATURES, strict=True)
        for attempt, (prompt, (num_ctx, num_predict), temperature) in enumerate(steps, 1):
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt.format(n=n_frames, style=style),
                        "images": images,
                    }
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
            }
            if profile and profile.expected:
                payload |= {"logprobs": True, "top_logprobs": _TOP_LOGPROBS}
            self.used = True
            try:
                resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=300)
                resp.raise_for_status()
                body = resp.json()
                content = body["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as err:
                logger.warning("Vision request failed: %s", err)
                return None
            result = _parse(content)
            if result is not None and profile and profile.expected:
                ev = _expected_score(body.get("logprobs"))
                if ev is not None:  # None: no logprobs (an older Ollama): the score as given
                    result = VisionResult(score=round(ev, 2), label=result.label)
            if result is not None:
                if attempt > 1:
                    logger.info("Vision retry %d succeeded", attempt)
                return result
            logger.info(
                "Vision reply %d empty/unparseable (done_reason=%s, %s tokens); %s",
                attempt,
                body.get("done_reason"),
                body.get("eval_count"),
                "retrying" if attempt < len(_ATTEMPTS) else "giving up",
            )
        logger.warning("Vision reply not parseable: %.120s", content)
        return None

    def unload(self) -> None:
        """Free the model's memory now (``keep_alive: 0``) instead of after Ollama's
        default 5 minutes; only if this rater sent it a request. Failures are not fatal."""
        if not self.used:
            return
        try:
            resp = requests.post(
                f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=10
            )
            resp.raise_for_status()
        except requests.RequestException as err:
            logger.debug("Could not unload Ollama model %s: %s", self.model, err)
            return
        self.used = False
        logger.debug("Unloaded Ollama model %s", self.model)


def _frame_times(start: float, end: float, n_frames: int) -> list[float]:
    """``n_frames`` evenly spaced inside [start, end], neither end included."""
    span = end - start
    return [start + span * (i + 1) / (n_frames + 1) for i in range(n_frames)]


def rating_key(model: str, style: str, times: list[float]) -> str:
    """Cache key of one rating: a hash of everything sent to the model except the frames
    (``RATING_VERSION``, model, style, both prompts, attempt options and temperatures, frame
    width and the model's :class:`RatingProfile` if it has one), then the frame times to the
    millisecond (the precision of the grab's seek). The Ollama URL is not part of it: the
    same model on another server rates the same. The model's digest is stored in the entry
    instead of hashed (:meth:`VisionRater._lookup`), so a rating can still be found when
    Ollama is down and the digest cannot be fetched."""
    profile = rating_profile(model)
    prompts = _prompts(style, profile)
    params: dict[str, Any] = {
        "version": RATING_VERSION,
        "model": model,
        "style": style,
        "prompts": list(prompts),
        "attempts": [list(a) for a in _ATTEMPTS],
        "temperatures": list(_TEMPERATURES),
        "width": FRAME_WIDTH,
    }
    if profile:  # only then, so the keys of the other models stay as they were
        params["profile"] = asdict(profile)
    return f"{cache.params_hash(params)}@" + ",".join(f"{t:.3f}" for t in times)


def _prompts(style: str, profile: RatingProfile | None) -> tuple[str, str]:
    """(first prompt, retry prompt) of ``style``, worded for one contact sheet when the
    profile sends one."""
    first, retry = _PROMPTS.get(style, _DEFAULT_PROMPTS)
    if not profile:
        return first, retry
    sheet = "This image is a grid of {n} frames, in reading order (left to right, top to bottom),"
    return (
        first.replace("These are {n} frames, in order,", sheet),
        retry.replace("{n} frames from", "A grid of {n} frames, in reading order, from", 1),
    )


def contact_sheet(frames: list[bytes], cols: int) -> bytes:
    """The JPEG ``frames`` on one JPEG, ``cols`` per row in reading order, each at the first
    frame's size (black where the last row is short). ValueError if one cannot be decoded."""
    tiles = []
    for data in frames:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("a frame could not be decoded")
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    tiles = [t if t.shape[:2] == (h, w) else cv2.resize(t, (w, h)) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_QUALITY])
    if not ok:
        raise ValueError("the contact sheet could not be encoded")
    return buf.tobytes()


def _expected_score(logprobs: object) -> float | None:
    """Expected score over the top logprobs of the token that starts the reply's score
    value; None if the reply has no logprobs, no digit there or nothing usable.

    Qwen writes a number digit by digit, so an alternative "1" may start "10": it counts as
    10 when the alternatives put more probability on 6-9 than on 0 and 2-5, else as 1 (from
    the probabilities, not the sampled digit, so a rerun scores the same). When "1" is the
    chosen token, it counts as 1 plus 9 times the chance that the next token is "0" (that
    token's "0" alternative, else 1 if the next token is "0")."""
    if not isinstance(logprobs, list):
        return None
    text = ""
    for i, tok in enumerate(logprobs):
        token = tok.get("token", "") if isinstance(tok, dict) else ""
        before, text = text, text + token
        if '"score"' not in before or ":" not in before.rsplit('"score"', 1)[-1]:
            continue
        if not token.strip():
            continue
        chosen, alts = token.strip(), tok.get("top_logprobs")
        if not _is_digit(chosen) or not isinstance(alts, list):
            return None
        digits = []
        for alt in alts:
            digit, lp = _token(alt), alt.get("logprob") if isinstance(alt, dict) else None
            if _is_digit(digit) and _usable_logprob(lp):
                digits.append((digit, math.exp(lp)))
        if chosen == "1":
            nxt = logprobs[i + 1] if i + 1 < len(logprobs) else None
            one = 1.0 + 9.0 * _zero_next(nxt)  # the exact chance of "10"
        else:
            high = sum(p for d, p in digits if d in "6789")
            low = sum(p for d, p in digits if d in "02345")
            one = 10.0 if high > low else 1.0
        total = sum(p for _, p in digits)
        weighted = sum(p * (one if d == "1" else float(d)) for d, p in digits)
        score = weighted / total if total else None
        return score if score is not None and math.isfinite(score) else None
    return None


def _is_digit(token: str) -> bool:
    return len(token) == 1 and token in "0123456789"


def _usable_logprob(lp: object) -> bool:
    """A real log probability: a finite number at most 0 (not a bool)."""
    if isinstance(lp, bool) or not isinstance(lp, (int, float)):
        return False
    return math.isfinite(lp) and lp <= 0


def _zero_next(entry: object) -> float:
    """Chance that the token after a chosen "1" is "0": its "0" alternative, else 1 if it is
    "0" and 0 if not."""
    alts = entry.get("top_logprobs") if isinstance(entry, dict) else None
    if isinstance(alts, list):
        for alt in alts:
            lp = alt.get("logprob") if isinstance(alt, dict) else None
            if _token(alt) == "0" and _usable_logprob(lp):
                return math.exp(lp)
        if alts:
            return 0.0
    return 1.0 if _token(entry) == "0" else 0.0


def _token(entry: object) -> str:
    """A logprobs entry's token, stripped ("" if it has none)."""
    token = entry.get("token") if isinstance(entry, dict) else None
    return token.strip() if isinstance(token, str) else ""


def _cached(entry: object) -> VisionResult | None:
    """A cache entry as a rating; None (a miss) if it is malformed."""
    if not isinstance(entry, dict):
        return None
    score, label = entry.get("score"), entry.get("label")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not isinstance(label, str):
        return None
    if not math.isfinite(score):
        return None
    return VisionResult(score=float(score), label=label)


def _json_object(content: str) -> object:
    """A reply's JSON from its first ``{`` to its last ``}``; None if it has none."""
    match = re.search(r"\{.*\}", content, re.S)
    return json.loads(match.group(0)) if match else None


def _parse(content: str) -> VisionResult | None:
    try:
        data = _json_object(content)
        if not isinstance(data, dict) or "score" not in data:
            return None
        score = float(data.get("score", 0))
    except (ValueError, TypeError):
        return None
    if math.isnan(score):
        return None
    score = max(0.0, min(10.0, score))
    return VisionResult(score=score, label=str(data.get("label", "")).strip())
