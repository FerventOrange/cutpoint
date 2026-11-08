"""Labels for the regions HUD discovery proposes, from the shared vision model.

Discovery's rules tell a kill feed from other text by where it is and how it changes, which
fails when a loadout box changes more than the feed, or when stream chat sits next to it. After
discovery, each proposal is shown to the vision model (the one ``cutpoint events`` and
``analyze --vision`` use: Ollama, or a hosted Vision model the user allowed) as up to
:data:`CROPS` crops of the proposal from frames where it had text, with its position in words,
and the model says what the area is used for (:data:`LABELS`). The model never draws a box: it
only names rectangles discovery found.

The labels are stored in the discovered preset (``discovery.labels``), so it runs once per game.
A label answer is cached per proposal in ``<discovery folder>/feed_labels.json``, keyed on the
model (provider, host and model id), the rectangle and the frames shown.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, progress
from cutpoint.errors import LowMemoryError
from cutpoint.events.discover import REF_H, REF_W, boxes_in, read_image

logger = logging.getLogger(__name__)

LABEL_VERSION = 2  # bump when the prompt, schema or crops change: cached labels are not reused
LABELS = ("kill_feed", "chat", "notice", "banner", "menu", "readout", "other")
CROPS = 4  # crops per call
MARGIN = 0.15  # crop margin, share of the proposal's size
MARGIN_MIN = 16  # ...and at least this many reference pixels
CROP_MIN_SIDE = 256  # crops are upscaled until the short side is this long...
CROP_MAX_SIDE = 1280  # ...but the long side stays within this
MAX_CALLS = 12  # proposals asked about per game
SKIP_KINDS = ("readout", "popup", "banner")  # the rules are confident about these
STOP_AFTER = 2  # failed calls in a row after which the rest are not asked
ASK_TIMEOUT = 60  # seconds a call may take: a stuck Ollama costs STOP_AFTER x this at most

SCHEMA = {
    "type": "object",
    "properties": {
        "text_in_box": {"type": "string"},
        "label": {"type": "string", "enum": list(LABELS)},
    },
    "required": ["text_in_box", "label"],
}
_DEFS = (
    "Labels:\n"
    "- kill_feed: the game's own list of recent kills/eliminations; rows of killer name, weapon "
    "icon or verb, victim name.\n"
    "- chat: messages typed by people (stream viewers or players): a name followed by a "
    "message.\n"
    "- notice: short game callouts or hints (objective, location name, status line, 'friendly "
    "fire active').\n"
    "- banner: large centred text announcing a phase or result (round won, action phase, "
    "1 vs 4).\n"
    "- menu: menu, lobby, settings or loadout screen text.\n"
    "- readout: numbers such as health, ammo, speed, score, timer, FPS.\n"
    "- other: anything else, or nothing there.\n"
)
_PROMPT = (
    "These are {n} crops of the same screen area ({where}) taken from different moments of a "
    "recorded game stream. Decide what that screen area is used for.\n"
    + _DEFS
    + "Describe briefly what text is in the crops, then give the label."
)

Labeler = Callable[[list[dict[str, Any]], list[dict[str, Any]], Path], dict[str, dict[str, str]]]


def where(p: dict[str, Any]) -> str:
    """The proposal's position in words: ``top-right of the screen, 337x53 px of 1920x1080``."""
    cx, cy = p["x"] + p["w"] / 2, p["y"] + p["h"] / 2
    v = "top" if cy < REF_H / 3 else "bottom" if cy > 2 * REF_H / 3 else "middle"
    h = "left" if cx < REF_W / 3 else "right" if cx > 2 * REF_W / 3 else "centre"
    return f"{v}-{h} of the screen, {p['w']}x{p['h']} px of {REF_W}x{REF_H}"


def pick_frames(
    frames: list[dict[str, Any]], p: dict[str, Any], n: int = CROPS
) -> list[dict[str, Any]]:
    """Up to ``n`` frames, spread over the footage, where the proposal ``p`` had text: of the
    frames with at least the median number of text boxes in it."""
    seen = [(f, len(boxes_in(f, p["x"], p["y"], p["w"], p["h"]))) for f in frames]
    seen = [(f, k) for f, k in seen if k]
    if not seen:
        return []
    median = float(np.median([k for _, k in seen]))
    rich = sorted((f for f, k in seen if k >= median), key=lambda f: (f["source"], f["t"]))
    if len(rich) <= n:
        return rich
    return [rich[round(i * (len(rich) - 1) / (n - 1))] for i in range(n)]


def crop_jpeg(img: np.ndarray, p: dict[str, Any]) -> bytes | None:
    """The JPEG of proposal ``p`` plus a margin, cut from a frame of any size (same aspect as
    1920x1080) and upscaled so the short side is :data:`CROP_MIN_SIDE` px."""
    import cv2  # noqa: PLC0415

    sx = img.shape[1] / REF_W
    mx, my = max(MARGIN_MIN, p["w"] * MARGIN), max(MARGIN_MIN, p["h"] * MARGIN)
    x0, y0 = max(0, int((p["x"] - mx) * sx)), max(0, int((p["y"] - my) * sx))
    x1 = min(img.shape[1], int((p["x"] + p["w"] + mx) * sx))
    y1 = min(img.shape[0], int((p["y"] + p["h"] + my) * sx))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    scale = max(1.0, CROP_MIN_SIDE / min(crop.shape[:2]))
    if max(crop.shape[:2]) * scale > CROP_MAX_SIDE:
        scale = min(scale, CROP_MAX_SIDE / max(crop.shape[:2]))
    if scale != 1.0:
        crop = cv2.resize(
            crop,
            (max(32, int(crop.shape[1] * scale)), max(32, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes() if ok else None


def candidates(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The proposals to ask about, best guesses for a kill feed first (feeds, notices outside
    the other regions, other notices, the rest), at most :data:`MAX_CALLS`. Readouts, banners
    and score popups are not asked about."""

    def rank(p: dict[str, Any]) -> int:
        if p["kind"] == "feed":
            return 0
        if p.get("evidence", {}).get("outside"):
            return 1
        return 2 if p["kind"] == "notice" else 3

    todo = [p for p in proposals if p["kind"] not in SKIP_KINDS]
    return sorted(todo, key=rank)[:MAX_CALLS]


def _key(model: str, p: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    return cache.params_hash(
        {
            "v": LABEL_VERSION,
            "model": model,
            "prompt": cache.params_hash({"p": _PROMPT, "s": SCHEMA}, 8),
            "box": [p["x"], p["y"], p["w"], p["h"]],
            "frames": [f["id"] for f in frames],
        },
        16,
    )


def label_proposals(
    proposals: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    frame_dir: Path,
    ask: Callable[[str, list[bytes], dict[str, Any]], dict[str, Any] | None],
    model: str,
) -> dict[str, dict[str, str]]:
    """Ask the vision model what each candidate proposal is.

    Args:
        proposals: Discovery's proposals (:func:`cutpoint.events.discover.cluster_regions`).
        frames: The gameplay frames discovery OCR'd.
        frame_dir: Folder of their JPEGs (``frames/``); the cache file is next to it.
        ask: ``(prompt, jpegs, schema) -> reply dict or None``.
        model: The model's cache key (provider, host and model id).

    Returns:
        ``{proposal name: {"label": ..., "model": ...}}`` for the proposals the model
        labelled.
    """

    path = frame_dir.parent / "feed_labels.json"
    stored = cache.load_json(path)
    store: dict[str, Any] = stored if isinstance(stored, dict) else {}
    images: dict[str, np.ndarray | None] = {}
    out: dict[str, dict[str, str]] = {}
    failed = 0
    for p in candidates(proposals):
        picked = pick_frames(frames, p)
        if not picked:
            continue
        key = _key(model, p, picked)
        hit = store.get(key)
        if isinstance(hit, dict) and hit.get("label") in LABELS:
            out[p["name"]] = {"label": hit["label"], "model": model}
            continue
        jpegs = []
        for f in picked:
            if f["frame"] not in images:
                images[f["frame"]] = read_image(frame_dir / f["frame"])
            img = images[f["frame"]]
            jpeg = crop_jpeg(img, p) if img is not None else None
            if jpeg:
                jpegs.append(jpeg)
        if not jpegs:
            continue
        reply = ask(_PROMPT.format(n=len(jpegs), where=where(p)), jpegs, SCHEMA)
        label = reply.get("label") if reply else None
        if label not in LABELS:
            failed += 1
            if failed >= STOP_AFTER:
                logger.info("Feed labels: the vision model gave no usable answer; stopping")
                break
            continue
        failed = 0
        # what the model read (names, chat) is not kept: only its label
        logger.debug("Feed label %s: %s (read %.80r)", p["name"], label, reply.get("text_in_box"))
        out[p["name"]] = {"label": label, "model": model}
        store[key] = {"label": label}
        cache.save_json(path, store)
    return out


def model_key(rater: Any) -> str:
    """The model's cache key, as the VLM layer's: ``provider@host:model`` (an Ollama's host is
    that of its URL)."""
    from cutpoint.assistant.providers import host_port  # noqa: PLC0415 (imports vision)

    hosted = rater.provider_name != "ollama"
    host = rater.host if hosted else host_port(rater.url)
    return f"{rater.provider_name}@{host}:{rater.model}"


def make_labeler(rater: Any) -> Labeler:
    """A :data:`Labeler` for ``discover_regions`` that asks the shared vision client
    ``rater`` (:class:`cutpoint.vision.VisionRater`).

    It asks nothing, and gives no labels, when the model is not there: an Ollama that does not
    answer or lacks the model (an INFO line), a provider the user has not allowed for Vision
    (the ``vision_consent_needed`` warning, :meth:`~cutpoint.vision.VisionRater.warn_consent`)
    or too little free memory (a WARNING). The rules then decide alone. The model is freed
    after the last call unless it is hosted.
    """
    state: dict[str, bool] = {}

    def usable() -> bool:
        if "ok" not in state:
            if rater.consent_needed:
                rater.warn_consent()
                state["ok"] = False
            else:
                state["ok"] = rater.available()
                if not state["ok"]:
                    logger.info("Feed labels: no vision model answers; the rules decide")
        return state["ok"]

    def ask(prompt: str, jpegs: list[bytes], schema: dict[str, Any]) -> dict[str, Any] | None:
        return rater.describe(
            prompt,
            jpegs,
            schema,
            num_ctx=16384,
            num_predict=500,
            temperature=0.0,
            timeout=ASK_TIMEOUT,
        )

    def label(
        proposals: list[dict[str, Any]], frames: list[dict[str, Any]], frame_dir: Path
    ) -> dict[str, dict[str, str]]:
        if not candidates(proposals) or not usable():
            return {}
        progress.update(detail="labelling HUD regions with the vision model")
        try:
            labels = label_proposals(proposals, frames, frame_dir, ask, model_key(rater))
        except LowMemoryError as err:
            logger.warning("Feed labels skipped, the rules decide: %s", err)
            return {}
        finally:
            if rater.provider_name == "ollama":
                rater.unload()
        logger.info(
            "Feed labels (%s): %s",
            model_key(rater),
            ", ".join(f"{n} {v['label']}" for n, v in labels.items()) or "none",
        )
        return labels

    return label
