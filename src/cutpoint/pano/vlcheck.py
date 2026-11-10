"""Framing check by a local vision model on a rendered preview (detection / QA only).

The geometric checks (:mod:`cutpoint.pano.qa_paths`) only see what the person
detector boxes: an outstretched arm, a jacket strip at the frame edge, a mitten
or goggles right at the lens, or a close-up of a sign as a cut-away destination
slip through. For a planned path and its low-res preview this module

1. picks a few frames: a sparse baseline (every ``every`` s) plus the path's
   suspect times (``vl_suspect`` from :func:`cutpoint.pano.director.compute_path`:
   near people in view, cut destinations), at most ``max_frames``; and
2. asks the default vision model (local Ollama, :mod:`cutpoint.vision`) one
   frame per request for strict JSON ``{"blocked", "where", "what", "low_info"}``
   (one retry with a short prompt when the reply does not parse).

The flags are stored (path ``vl`` record, ``vl_check.json``) and counted by
``360 qa``, which fails a clip with more than 1 bad frame in its span; the
director only reports them, they do not drive it. A closed loop (flags as grid
penalties, then as safe-area obstacles, re-plan kept only if it checks better)
was measured on 36 segments and fixed too few of the problem frames for its cost.

The prompt names the activity, its gear, scenery and low-information frames from
the activity profile's ``vl`` wording (:func:`prompts_for`; the active prompt set when
the checker is made); its hash is part of the cache key.

One frame per request: qwen3-vl:8b (a thinking variant, the default vision
model when this was measured) reasons even with ``think=false``; a multi-image
request used up its token budget before the answer. Replies are cached per
(clip, time, view) under the cache folder.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cutpoint import cache, errors, ffmpeg_utils, state
from cutpoint.pano.qa import nearest_keys

if TYPE_CHECKING:
    from cutpoint.activity import Prompts

logger = logging.getLogger(__name__)

VL_VERSION = 2  # bump when the prompt or the reply schema changes (cache key)
WHERES = ("left", "right", "center", "bottom", "top", "none")
_WHERE_ALIASES = {
    "centre": "center",
    "middle": "center",
    "bottom-left": "left",
    "bottom-right": "right",
    "top-left": "left",
    "top-right": "right",
}

# The profile's ``vl`` wording fills {video}, {gear}, {scenery} and {low_info}
# (:func:`prompts_for`); {Video} is {video} capitalised.
PROMPT_TEMPLATE = (
    "One frame from a first-person {video} video (a virtual camera aimed inside a 360 video). "
    "Decide: blocked = something very close to the lens covers part of the view (a hand, "
    "arm, glove or mitten, {gear}, a shoulder or jacket strip at the frame edge, goggles "
    "or a face filling the frame). The friend the camera is framing is the subject, NOT "
    "blocked, even close up or wearing goggles, unless the frame edge cuts through them or "
    "their hand or pole reaches at the lens. People at a normal distance, {scenery} "
    "are NOT blocked. low_info = the frame shows nothing worth watching (a close-up of a "
    "sign, board or wall, {low_info}, heavy blur). Answer immediately, no reasoning, "
    'JSON only: {{"blocked": true|false, "where": "left|right|center|bottom|top|none", '
    '"what": "<2-5 words>", "low_info": true|false}}'
)
# Retry: shorter, no reasoning (the pattern of cutpoint.vision's retry prompt).
RETRY_TEMPLATE = (
    "{Video} video frame. Answer immediately, no reasoning. blocked = a hand, arm, glove, pole, "
    "goggles or a jacket edge right at the lens. low_info = a sign close-up, {low_info}, "
    'or blur. Output only JSON: {{"blocked": true|false, "where": '
    '"left|right|center|bottom|top|none", "what": "<2-5 words>", "low_info": true|false}}'
)
CHECK_NUM_CTX = 8192  # the context one frame needs; a larger loaded one is reused (#865)
NUM_PREDICT = (2048, 3072)  # per attempt: the prompt, then the retry


def prompts_for(p: Prompts) -> tuple[str, str]:
    """``(prompt, retry prompt)`` in the ``vl`` wording of a prompt set."""
    vl = dict(p.vl)
    words = {k: vl.get(k, "") for k in ("video", "gear", "scenery", "low_info")}
    words["Video"] = words["video"][:1].upper() + words["video"][1:]
    return PROMPT_TEMPLATE.format(**words), RETRY_TEMPLATE.format(**words)


def wording_id(p: Prompts) -> str:
    """Cache-key part of a prompt set's wording: a short hash of it."""
    return cache.params_hash(prompts_for(p), 8)


# clip id -> {cache key: reply}; the on-disk cache is the source of truth
_MEMO: dict[str, dict[str, Any]] = state.register_container(__name__, "_MEMO", {})


@dataclass(frozen=True)
class VLFlag:
    """The model's verdict on one preview frame (``t`` in clip-source seconds)."""

    t: float
    blocked: bool
    where: str
    what: str
    low_info: bool

    @property
    def bad(self) -> bool:
        return self.blocked or self.low_info


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object in ``text`` that carries a verdict, in order."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(d, dict) and ("blocked" in d or "low_info" in d):
            out.append(d)
    return out


def parse_reply(content: str, thinking: str = "") -> dict[str, Any] | None:
    """The verdict in a reply: the first in ``content``, else the last in ``thinking``.

    The thinking fallback covers a reply cut off after the model already wrote
    its answer in the reasoning. ``where`` is normalised to :data:`WHERES` (a
    blocked frame without a usable ``where`` counts as ``center``).
    """
    found = _objects(content)
    d = found[0] if found else (_objects(thinking) or [None])[-1]
    if d is None:
        return None
    blocked = _as_bool(d.get("blocked", False))
    where = str(d.get("where", "none")).strip().lower()
    where = _WHERE_ALIASES.get(where, where)
    if where not in WHERES or (blocked and where == "none"):
        where = "center"
    return {
        "blocked": blocked,
        "where": where if blocked else "none",
        "what": str(d.get("what", "")).strip()[:60],
        "low_info": _as_bool(d.get("low_info", False)),
    }


class VLChecker:
    """One-frame requests to the configured Vision role (:class:`cutpoint.vision.VisionRater`)."""

    def __init__(
        self, url: str | None = None, model: str | None = None, prompts: Prompts | None = None
    ) -> None:
        from cutpoint.activity import active_prompts  # noqa: PLC0415
        from cutpoint.vision import VisionRater  # noqa: PLC0415

        p = prompts or active_prompts()
        self.attempts = tuple(zip(prompts_for(p), NUM_PREDICT, strict=True))
        self.wording = wording_id(p)
        self.rater = VisionRater(url, model)
        self.model = self.rater.model
        self.provider_name = getattr(self.rater, "provider_name", "ollama")
        # the frame cache's key: a hosted provider's endpoint, an Ollama's host[:port]
        self.host = getattr(self.rater, "host", "")
        self.dead = False  # set when the server stops answering: no more requests

    def available(self) -> bool:
        if not self.rater.available():
            return False
        self.model = self.rater.model  # resolved by the check (the frame cache's key)
        return True

    def unload(self) -> None:
        """Free the model's VRAM after the last check (:meth:`VisionRater.unload`)."""
        self.rater.unload()

    def _post(self, prompt: str, image: str, num_predict: int) -> dict[str, Any] | None:
        import requests  # noqa: PLC0415

        from cutpoint.vision import refuse_redirect  # noqa: PLC0415

        # a provider the user did not agree to (a hosted one, or an Ollama on another machine):
        # no request, only the consent warning
        if self.rater.consent_needed:
            self.rater.warn_consent()
            self.dead = True
            return None
        if self.rater.via:  # the frame is a JPEG; the prompt asks for JSON, no sampling options
            text = self.rater.via.ask(prompt, [base64.b64decode(image)])
            if text is None:
                logger.warning("VL check request failed; no more VL checks")
                self.dead = True
                return None
            return {"content": text}

        # the context the model is loaded with already when larger (no reload, #732/#865)
        num_ctx = self.rater._context(CHECK_NUM_CTX)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [image]}],
            "stream": False,
            "think": False,
            # no "format": the JSON grammar made this model pad with whitespace
            "options": {"temperature": 0.1, "num_ctx": num_ctx, "num_predict": num_predict},
        }
        self.rater.check_memory(num_ctx)  # raises LowMemoryError: the run stops (resumable)
        self.rater.used = True  # so unload() frees it
        try:
            resp = requests.post(
                f"{self.rater.url}/api/chat", json=payload, timeout=(3, 300), allow_redirects=False
            )
            refuse_redirect(resp)
            resp.raise_for_status()
            return resp.json().get("message") or {}
        except requests.RequestException as err:
            logger.warning("VL check request failed: %s; no more VL checks", err)
            self.dead = True
            return None
        except ValueError as err:
            logger.warning("VL check reply is not JSON: %s", err)
            return None

    def ask(self, jpeg: bytes) -> dict[str, Any] | None:
        """The verdict for one JPEG frame (one retry), or None when both attempts fail
        or the server is :attr:`dead`."""
        if self.dead:
            return None
        image = base64.b64encode(jpeg).decode("ascii")
        for prompt, num_predict in self.attempts:
            msg = self._post(prompt, image, num_predict)
            if msg is None:
                return None
            reply = parse_reply(str(msg.get("content") or ""), str(msg.get("thinking") or ""))
            if reply is not None:
                return reply
        logger.info("VL check reply not parseable after a retry")
        return None


def sample_times(path: dict[str, Any], every: float = 3.0, max_frames: int = 4) -> np.ndarray:
    """Preview times to check, at most ``max_frames``: a baseline spread over the whole
    segment (about every ``every`` s, capped so up to half the frames stay free for
    suspects), then the path's ``vl_suspect`` times (at least 1 s from a chosen frame,
    the one farthest from the chosen frames and the segment ends first); with no such
    suspect left, the middle of the largest gap (segment ends included) while that gap
    is at least ``1.5 * every``."""
    start, end = float(path["segment"]["start"]), float(path["segment"]["end"])
    dur = end - start
    sus = [float(s) - start for s in path.get("vl_suspect", [])]
    sus = [t for t in sus if 0.0 < t < dur - 0.1]
    n_base = max(1, min(round(dur / every), max_frames - min(len(sus), max_frames // 2)))
    out = list((np.arange(n_base) + 0.5) * dur / n_base)
    while len(out) < max_frames:
        free = [t for t in sus if min(abs(t - u) for u in out) >= 1.0]
        if free:
            pick = max(free, key=lambda t: min(abs(t - u) for u in (0.0, *out, dur)))
        else:
            _, a, b = max((b - a, a, b) for a, b in pairwise([0.0, *sorted(out), dur]))
            if b - a < 1.5 * every:
                break
            pick = (a + b) / 2
        out.append(pick)
    return np.array(sorted(out))


def _cache_file(clip_id: str) -> Path:
    return cache.cache_root() / "pano_vl" / f"{clip_id}.json"


def frame_key(
    path: dict[str, Any],
    t: float,
    model: str,
    provider: str = "",
    wording: str = "",
    host: str = "",
) -> str:
    """Cache key of the preview frame at source time ``t``: what the view shows there
    (``wording``: :func:`wording_id`; the ``provider`` with its ``host``, so one model id at
    two servers checks apart; Ollama without a host keeps the old key)."""
    k = nearest_keys(path["keys"], [t])[0]
    vert = "v" if path.get("vertical") else "h"
    prov = f"|{provider}" if provider and (provider != "ollama" or host) else ""
    prov += f"@{host}" if prov and host else ""
    prov += f"|w{wording}" if wording else ""
    return (
        f"{VL_VERSION}|{model}{prov}|{vert}|{t:.2f}|"
        f"{float(k['yaw']) % 360:.0f}|{float(k['pitch']):.0f}|{float(k['fov']):.0f}"
    )


def check_preview(
    checker: VLChecker,
    preview: Path,
    path: dict[str, Any],
    *,
    every: float = 3.0,
    max_frames: int = 4,
) -> tuple[list[VLFlag], dict[str, Any]]:
    """Ask the model about the sampled frames of ``preview`` (rendered from ``path``).

    Returns:
        The flags (frames the model answered) and a record: ``frames``, ``bad``,
        ``bad_share``, ``seconds`` (model time; cache hits are free), ``cached``,
        ``flags`` (the bad ones).
    """
    start = float(path["segment"]["start"])
    clip_id = str(path.get("clip", "clip"))
    if clip_id not in _MEMO:
        _MEMO[clip_id] = cache.load_json(_cache_file(clip_id)) or {}
    store = _MEMO[clip_id]
    tps = sample_times(path, every, max_frames)
    flags: list[VLFlag] = []
    secs, hits, dirty = 0.0, 0, False
    for tp in tps:
        t = start + float(tp)
        key = frame_key(
            path,
            t,
            checker.model,
            provider=getattr(checker, "provider_name", ""),
            wording=getattr(checker, "wording", ""),
            host=getattr(checker, "host", ""),
        )
        reply = store.get(key)
        if reply is None:
            try:
                grabs = ffmpeg_utils.grab_images(preview, float(tp), count=1, width=640, quality=4)
                jpeg = grabs[0]
            except errors.FFmpegError as err:
                logger.warning("VL check frame %s @%.2f: %s", preview.name, tp, err)
                continue
            t_ask = time.perf_counter()
            reply = checker.ask(jpeg)
            secs += time.perf_counter() - t_ask
            if reply is None:
                continue
            store[key] = reply
            dirty = True
        else:
            hits += 1
        flags.append(VLFlag(t=round(t, 3), **reply))
    if dirty:
        _cache_file(clip_id).parent.mkdir(parents=True, exist_ok=True)
        cache.save_json(_cache_file(clip_id), store, indent=0)
    bad = sum(f.bad for f in flags)
    return flags, {
        "frames": len(flags),
        "bad": bad,
        "bad_share": round(bad / len(flags), 3) if flags else 0.0,
        "seconds": round(secs, 1),
        "cached": hits,
        "flags": [asdict(f) for f in flags if f.bad],
    }
