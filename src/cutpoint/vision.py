"""Optional frame rating with a local Ollama vision model (e.g. qwen3-vl:8b)."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from cutpoint import paths
from cutpoint.backends import ollama_url
from cutpoint.ffmpeg_utils import FFmpegError, grab_frame_jpeg

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "qwen3-vl:8b"


def default_url() -> str:
    """Ollama URL (see :func:`cutpoint.backends.ollama_url`)."""
    return ollama_url()


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
# (prompt, num_ctx, num_predict) per attempt. 3 frames at 640 px are ~3300
# prompt tokens, so num_ctx must leave room for the reasoning plus the JSON.
_ATTEMPTS = ((_PROMPT, 8192, 3072), (_RETRY_PROMPT, 12288, 6144))


@dataclass
class VisionResult:
    """Rating returned by the vision model."""

    score: float
    label: str


class VisionRater:
    """Client for Ollama's /api/chat endpoint with images."""

    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or default_url()).rstrip("/")
        if not self.url.startswith("http"):
            self.url = f"http://{self.url}"
        self.model = model or default_model()

    def available(self) -> bool:
        """True if the server answers and has the model pulled."""
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
        try:
            resp = requests.post(f"{self.url}/api/{endpoint}", json=payload, timeout=600)
            resp.raise_for_status()
            body = resp.json()
            content = body["response"] if raw else body["message"]["content"]
            match = re.search(r"\{.*\}", content, re.S)
            data = json.loads(match.group(0)) if match else None
        except (requests.RequestException, KeyError, ValueError) as err:
            logger.warning("Ollama request failed: %s", err)
            return None
        return data if isinstance(data, dict) else None

    def rate(
        self, src: Path, start: float, end: float, style: str, n_frames: int = 3
    ) -> VisionResult | None:
        """Grab ``n_frames`` across [start, end] and ask the model for a rating."""
        span = end - start
        times = [start + span * (i + 1) / (n_frames + 1) for i in range(n_frames)]
        try:
            images = [base64.b64encode(grab_frame_jpeg(src, t)).decode("ascii") for t in times]
        except FFmpegError as err:
            logger.warning("Vision frame grab failed: %s", err)
            return None
        content = ""
        for attempt, (prompt, num_ctx, num_predict) in enumerate(_ATTEMPTS, 1):
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            _PROMPT_SKI if style == "ski" and attempt == 1 else prompt
                        ).format(n=n_frames, style=style),
                        "images": images,
                    }
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0.2 if attempt == 1 else 0.4,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
            }
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


def _parse(content: str) -> VisionResult | None:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if not isinstance(data, dict) or "score" not in data:
            return None
        score = max(0.0, min(10.0, float(data.get("score", 0))))
    except (ValueError, TypeError):
        return None
    return VisionResult(score=score, label=str(data.get("label", "")).strip())
