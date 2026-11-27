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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, progress
from cutpoint.errors import LowMemoryError
from cutpoint.events.discover import REF_H, REF_W, boxes_in
from cutpoint.imageio import read_image

logger = logging.getLogger(__name__)

LABEL_VERSION = 2  # bump when the prompt, schema or crops change: cached labels are not reused
LABELS = ("kill_feed", "chat", "notice", "banner", "menu", "readout", "other")
CROPS = 4  # crops per call
MARGIN = 0.15  # crop margin, share of the proposal's size
MARGIN_MIN = 16  # ...and at least this many reference pixels
CROP_MIN_SIDE = 256  # crops are upscaled until the short side is this long...
CROP_MAX_SIDE = 1280  # ...but the long side stays within this
MAX_CALLS = 12  # proposals asked about per game
SKIP_KINDS = ("readout", "popup", "banner", "score")  # the rules are confident about these
STOP_AFTER = 2  # failed calls in a row after which the rest are not asked
ASK_TIMEOUT = 60  # seconds a call may take
# ...but loading the model into an Ollama (#796) is a request of its own before the first call
# (VisionRater.load), and gets as long as one of the VLM layer's requests
# (VisionRater._rate_frames) or a local AI Chat read (CHAT_TIMEOUT_S). An Ollama that refuses
# that request gives the first call this timeout instead, as it then loads the model
FIRST_ASK_TIMEOUT = 300
LABEL_NUM_CTX = 16384  # the load and every call ask for the same context (no reload)
# all the calls of one discovery together (the model load not counted): a slow model labels the
# best guesses for a kill feed (candidates() asks them first) and the rules decide the rest.
# A call that runs out of its timeout stops the labelling: the next one would too (#845)
LABEL_BUDGET = 120
MIN_ASK = 15  # seconds of budget a call needs at least; with less left nothing more is asked


class StopLabelling(Exception):
    """Raised by a :data:`Labeler`'s ``ask`` when no more calls should go out (the model is
    too slow, or the budget is spent); the labels so far are kept."""


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
    """The JPEG of proposal ``p`` plus a margin, cut from a frame of any size (x and y scale
    on their own) and upscaled so the short side is :data:`CROP_MIN_SIDE` px."""
    import cv2  # noqa: PLC0415

    sx, sy = img.shape[1] / REF_W, img.shape[0] / REF_H
    mx, my = max(MARGIN_MIN, p["w"] * MARGIN), max(MARGIN_MIN, p["h"] * MARGIN)
    x0, y0 = max(0, int((p["x"] - mx) * sx)), max(0, int((p["y"] - my) * sy))
    x1 = min(img.shape[1], int((p["x"] + p["w"] + mx) * sx))
    y1 = min(img.shape[0], int((p["y"] + p["h"] + my) * sy))
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
    the other regions, other notices, the rest), at most :data:`MAX_CALLS`. Readouts, banners,
    score popups and score readouts are not asked about."""

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
        model: The model's cache key (provider, host and model id). ``ask`` may raise
            :class:`StopLabelling` to end the labelling; the labels so far are kept.

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
        try:
            reply = ask(_PROMPT.format(n=len(jpegs), where=where(p)), jpegs, SCHEMA)
        except StopLabelling as why:
            logger.info("Feed labels: %s; the rules decide the rest", why)
            break
        label = reply.get("label") if reply else None
        if label not in LABELS:
            failed += 1
            if failed >= STOP_AFTER:
                logger.info("Feed labels: the vision model gave no usable answer; stopping")
                break
            continue
        failed = 0
        # what the model read (names, chat) is not kept: only its label
        logger.debug("Feed label %s: %s", p["name"], label)
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


class VisionCalls:
    """The shared vision client's calls during one run's discoveries: the vision HUD map
    (:mod:`cutpoint.events.hud_map`) and the feed labels ask through one of these, so they share
    the model check, the model load and the budget of a game.

    It asks nothing when the model is not there: an Ollama that does not answer or lacks the
    model (an INFO line), a provider the user has not allowed for Vision (the
    ``vision_consent_needed`` warning, :meth:`~cutpoint.vision.VisionRater.warn_consent`). An
    Ollama model is loaded by a request of its own first; when ``/api/ps`` shows it on the CPU
    alone nothing is asked: it takes minutes per call (#845: 4 crops, about 3 min on a
    4-core CPU), so the rules decide. The calls of one game share a budget
    (:meth:`begin`), and one that times out ends the calls of that game (:class:`StopLabelling`);
    a model that loads on the CPU alone or does not load in time ends those of the run.
    """

    def __init__(self, rater: Any) -> None:
        self.rater = rater
        self.ok: bool | None = None  # rater usable, asked once
        self.ollama = rater.provider_name == "ollama"
        self.stop: list[str] = []  # why the model cannot answer: the same for every game
        self.deadline: list[float] = []  # set after a game's first call: the budget from then
        self.budget: float = LABEL_BUDGET
        self.carry: object = None  # the game whose next begin() keeps the budget (carry_over)
        self.cold = False  # the load request was refused: the next call loads the model

    def usable(self) -> bool:
        """Whether the model answers (asked once per run)."""
        if self.ok is None:
            if self.rater.consent_needed:
                self.rater.warn_consent()
                self.ok = False
            else:
                self.ok = bool(self.rater.available())
                if not self.ok:
                    logger.info("No vision model answers; the rules decide")
        return self.ok

    def begin(self, game: object, budget: float) -> None:
        """Start the calls of ``game`` (its frame folder) with ``budget`` seconds, unless the
        calls before handed their budget on to it (:meth:`carry_over`): the HUD map and the
        feed labels of one discovery share one budget."""
        if self.carry is not None and self.carry == game:
            self.carry = None
            return
        self.carry = None
        self.budget = budget
        self.deadline.clear()

    def carry_over(self, game: object) -> None:
        """Let the next :meth:`begin` for ``game`` go on with what is left of this budget."""
        self.carry = game

    def _warm(self) -> None:
        """Load an Ollama model before its first call, with its own long timeout (#796), so
        the calls then get ASK_TIMEOUT. Raises :class:`StopLabelling` when no call should go
        out (it runs on the CPU alone, or did not load in time)."""
        rater = self.rater
        if self.stop:
            raise StopLabelling(self.stop[0])
        if self.ollama and not rater.used:
            if rater.load(FIRST_ASK_TIMEOUT, num_ctx=LABEL_NUM_CTX):
                if rater.on_gpu() is False:
                    self.stop.append(f"{rater.model} runs on the CPU only, too slow to label")
                    raise StopLabelling(self.stop[0])
            elif rater.timed_out:
                self.stop.append(f"the vision model did not load within {FIRST_ASK_TIMEOUT}s")
                raise StopLabelling(self.stop[0])
            else:
                self.cold = True  # the load request was refused: the call loads it, as before

    def ready(self) -> bool:
        """Whether calls can go out now: the model answers (:meth:`usable`) and an Ollama model
        is loaded on a GPU. Checked before work that only the calls make worth doing (the HUD
        map top-up's discovery); :attr:`stop` says why not."""
        if not self.usable():
            return False
        try:
            self._warm()
        except StopLabelling:
            return False
        return True

    def ask(
        self, prompt: str, jpegs: list[bytes], schema: dict[str, Any], num_predict: int = 500
    ) -> dict[str, Any] | None:
        """One call: the reply dict or None. Raises :class:`StopLabelling` when no more calls
        should go out."""
        rater = self.rater
        self._warm()
        cold, self.cold = self.cold, False  # this call also loads the model
        if cold:
            timeout: float = FIRST_ASK_TIMEOUT
        else:
            if not self.deadline:
                self.deadline.append(time.monotonic() + self.budget)
            left = self.deadline[0] - time.monotonic()
            if left < MIN_ASK:
                raise StopLabelling(f"the {self.budget:.0f}s for the vision model are spent")
            timeout = min(ASK_TIMEOUT, left)
        reply = rater.describe(
            prompt,
            jpegs,
            schema,
            num_ctx=LABEL_NUM_CTX,
            num_predict=num_predict,
            temperature=0.0,
            timeout=timeout,
        )
        if reply is None and getattr(rater, "timed_out", False) is True:
            raise StopLabelling(f"a call ran out of its {timeout:.0f}s")
        return reply

    def release(self) -> None:
        """Free an Ollama model after the last call of a game (a hosted one stays); a budget
        handed on (:meth:`carry_over`) ends with it."""
        self.carry = None
        if self.ollama:
            self.rater.unload()


def make_labeler(rater: Any, calls: VisionCalls | None = None) -> Labeler:
    """A :data:`Labeler` for ``discover_regions`` that asks the shared vision client
    ``rater`` (:class:`cutpoint.vision.VisionRater`) through ``calls`` (a new
    :class:`VisionCalls` when None; the vision HUD map shares one with it).

    It asks nothing, and gives no labels, when the model is not there (see
    :class:`VisionCalls`) or with too little free memory (a WARNING). The rules then decide
    alone. The calls of one game share :data:`LABEL_BUDGET` (with the HUD map's, when it ran
    first), and one that times out ends the labelling (an INFO line); the labels so far are
    kept. The model is freed after the last call unless it is hosted.
    """
    calls = calls or VisionCalls(rater)

    def label(
        proposals: list[dict[str, Any]], frames: list[dict[str, Any]], frame_dir: Path
    ) -> dict[str, dict[str, str]]:
        if not candidates(proposals):
            calls.release()  # loaded for the HUD map: freed all the same
            return {}
        if not calls.usable():
            return {}
        calls.begin(frame_dir, LABEL_BUDGET)  # each game has its own budget
        progress.update(detail="labelling HUD regions with the vision model")
        try:
            labels = label_proposals(proposals, frames, frame_dir, calls.ask, model_key(rater))
        except LowMemoryError as err:
            logger.warning("Feed labels skipped, the rules decide: %s", err)
            return {}
        finally:
            calls.release()
        logger.info(
            "Feed labels (%s): %s",
            model_key(rater),
            ", ".join(f"{n} {v['label']}" for n, v in labels.items()) or "none",
        )
        return labels

    return label
