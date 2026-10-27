"""Optional frame rating with a local Ollama vision model (e.g. qwen3-vl:8b)."""

from __future__ import annotations

import base64
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from cutpoint import cache, paths
from cutpoint.backends import ollama_url
from cutpoint.ffmpeg_utils import FFmpegError, grab_frame_jpeg

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "qwen3-vl:8b"


def default_model() -> str:
    """Vision model: ``CUTPOINT_VISION_MODEL`` or qwen3-vl:8b."""
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
# Ratings cached per source (cache.source_dir), one entry per rating input (rating_key),
# stamped with the model's digest (VisionRater._lookup says how the digest is matched).
RATINGS_FILE = "vision-ratings.json"
# Bump when the rating input or the reply parsing changes outside what rating_key hashes
# (the prompts, attempt options and temperatures, frame width, model, style and frame times).
RATING_VERSION = 1


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
        ``cache_only`` a miss gives None (counted in ``missed``) and nothing is sent.
        """
        times = _frame_times(start, end, n_frames)
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
        key = rating_key(self.model, style, _frame_times(start, end, n_frames))
        return self._lookup(self._store(src), key)

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
        try:
            images = [
                base64.b64encode(grab_frame_jpeg(src, t, width=FRAME_WIDTH)).decode("ascii")
                for t in times
            ]
        except FFmpegError as err:
            logger.warning("Vision frame grab failed: %s", err)
            return None
        self.asked += 1
        content = ""
        steps = zip(_PROMPTS.get(style, _DEFAULT_PROMPTS), _ATTEMPTS, _TEMPERATURES, strict=True)
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
    width), then the frame times to the millisecond (the precision of the grab's seek). The
    Ollama URL is not part of it: the same model on another server rates the same. The
    model's digest is stored in the entry instead of hashed (:meth:`VisionRater._lookup`),
    so a rating can still be found when Ollama is down and the digest cannot be fetched."""
    prompts = _PROMPTS.get(style, _DEFAULT_PROMPTS)
    params = {
        "version": RATING_VERSION,
        "model": model,
        "style": style,
        "prompts": list(prompts),
        "attempts": [list(a) for a in _ATTEMPTS],
        "temperatures": list(_TEMPERATURES),
        "width": FRAME_WIDTH,
    }
    return f"{cache.params_hash(params)}@" + ",".join(f"{t:.3f}" for t in times)


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
