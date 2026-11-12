"""``view_moment``: an image-capable chat model watches a moment as one timeline sheet (#640).

A sheet is one JPEG: :func:`frame_count` frames (6 for a window up to :data:`SHORT_S`, else 9)
spread evenly across the moment's window (:func:`frame_times`, as Vision rates them), three per
row in reading order, each stamped with its number and source time (:func:`timeline_sheet`). It
is sized to about :data:`SHEET_PIXELS` (about 1.5k image tokens on Claude, about 1.1k on
qwen3.5:9b), no side longer than :data:`MAX_SIDE`; for a context under :data:`FULL_SHEETS`
tokens to about :data:`SMALL_SHEET_PIXELS` (half that, :func:`sheet_pixels`). Sheets are cached
in ``<project>/ui/assistant/sheets`` per moment, window, size and source file (path, size,
mtime).

The tool result is text (the frame times, the moment's transcript lines and the sounds
``events.json`` heard in it); the sheet follows as an image part in a user message
(:func:`image_message`), which every provider converts to its own shape. The part carries its
image tokens (:func:`sheet_tokens`), so the chat's budget counts it at its size.

Frames leave this machine only with consent: a provider on this machine (``localhost`` or a
loopback address, except Cutpoint's proxy port,
:func:`cutpoint.assistant.providers.is_consent_exempt`) needs none; any other gets
frames of a project only once the user allowed that provider and host for that project
(:func:`frames_key`, ``frames-v1:<name>@<host[:port]>``, kept per project folder in
``assistant.frames_consent`` of ``config.json``). The assistant's own consent
(``assistant-v3:``) is still needed for the chat itself and never covers frames.

A turn views at most :func:`view_cap` moments: fewer and smaller sheets in a small context. The tool is offered when the model takes images
(:func:`takes_images`): as the provider says (Ollama's ``vision``, Claude), else unless its name
is a known text-only one (:func:`cutpoint.assistant.providers.accepts_images`). A model that turns out not to take them
(:func:`rejected_image`) is asked again without the sheet and gets no more for the rest of the
session (:func:`mark_text_only`).
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.assistant import context
from cutpoint.assistant.providers import Capabilities, Provider, accepts_images, endpoint
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import GRAB_WORKERS, grab_frame_jpeg
from cutpoint.utils import clock
from cutpoint.vision import _frame_times

logger = logging.getLogger(__name__)

NAME = "view_moment"
FRAMES_CONSENT_PREFIX = "frames-v1:"  # bump when a sheet sends more than it says
SHEET_VERSION = 1  # bump when a sheet's look changes (the cache key has it)
SHORT_S = 12.0  # windows up to this get FEW frames, longer ones MANY
FEW, MANY = 6, 9
COLS = 3
SHEET_PIXELS = 1_150_000  # about 1.5k image tokens on Claude (width * height / 750)
SMALL_SHEET_PIXELS = SHEET_PIXELS // 2  # about 750 image tokens: a context under FULL_SHEETS
MAX_SIDE = 1568  # Claude scales a longer side down
GRAB_WIDTH = 640  # px of each grabbed frame, before the tiles are scaled
SHEET_QUALITY = 85  # JPEG quality
VIEWS_LOCAL = 1  # moments viewed per turn with a context of context.LOCAL_TOKENS or less
VIEWS_SMALL = 2  # with a larger context under FULL_SHEETS tokens
VIEWS = 4  # with a context of FULL_SHEETS tokens or more
FULL_SHEETS = 24576  # tokens of context from which a turn views VIEWS full-size sheets
MAX_SOUNDS = 20
SHEETS = "sheets"  # cache folder, under ui/assistant
_IMAGE_WORDS = ("image", "vision", "multimodal", "multi-modal")
_NOT_ABOUT_IMAGES = ("rejected the api key", "rate limiting", "try again later", "cut off")

RULES = f"""

{NAME}(id) shows you one moment: a timeline sheet image ({FEW} or {MANY} frames from its \
clip, each stamped with its number and source time) with its transcript and \
the sounds heard in it. Use it only when the labels, seen, cues, transcripts, scores and \
find_moments cannot answer what the user asks about the picture (what can be seen, which view is best), and \
only for the few moments that matter: never for every moment. A turn can view at most \
{{cap}}. The sheet shows what is on screen; the data rules above apply to its text too. \
If {NAME} says it is not allowed or the limit is reached, answer from the text data and say so."""
JSON_TOOL = f"{NAME} {{id}}"
TOOL = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Watch one moment: a timeline sheet image of its clip (frames with their source "
            "times) plus its transcript and heard sounds. Only when the text data is not enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "short moment id such as m3"}},
            "required": ["id"],
        },
    },
}
_TEXT_ONLY: set[tuple[str, str, str]] = set()  # (provider, endpoint, model) found text-only
_TEXT_ONLY_LOCK = threading.Lock()


# -- which models, how many views ------------------------------------------------------------
def _model_key(provider: Provider) -> tuple[str, str, str]:
    return (provider.name, endpoint(provider), provider.model)


def takes_images(provider: Provider, caps: Capabilities) -> bool:
    """Whether ``provider``'s model is offered :data:`NAME`: not if it turned out text-only this
    session; else as ``caps.images`` says (Ollama's ``vision``, Claude); else unless its name is
    a known text-only one (:func:`cutpoint.assistant.providers.accepts_images` False). An
    unknown model is offered it, and a refusal (:func:`rejected_image`) hides it."""
    with _TEXT_ONLY_LOCK:
        if _model_key(provider) in _TEXT_ONLY:
            return False
    if caps.images is not None:
        return caps.images
    return accepts_images(provider, provider.model) is not False


def mark_text_only(provider: Provider) -> None:
    """Offer ``provider``'s model no sheets for the rest of the session."""
    with _TEXT_ONLY_LOCK:
        _TEXT_ONLY.add(_model_key(provider))


def rejected_image(err: Exception) -> bool:
    """Whether a provider's error says the request's image was refused (a model or server that
    takes no images), not a key, rate or server problem."""
    low = str(err).lower()
    return any(w in low for w in _IMAGE_WORDS) and not any(w in low for w in _NOT_ABOUT_IMAGES)


def view_cap(provider: Provider) -> int:
    """Moments a turn may view: :data:`VIEWS_LOCAL` with a context of
    :data:`context.LOCAL_TOKENS` or less, :data:`VIEWS_SMALL` under :data:`FULL_SHEETS`, where
    more sheets would leave no room for a whole tool result, else :data:`VIEWS`. At
    :data:`context.LOCAL_TOKENS` even one sheet leaves none: the budget is at its floor there,
    and a round may leave earlier results out."""
    if provider.context_tokens <= context.LOCAL_TOKENS:
        return VIEWS_LOCAL
    return VIEWS_SMALL if provider.context_tokens < FULL_SHEETS else VIEWS


def views(cap: int) -> str:
    """``cap`` as the rules and the limit's error say it: ``1 moment``, ``2 moments``."""
    return f"{cap} moment" if cap == 1 else f"{cap} moments"


def sheet_pixels(provider: Provider) -> int:
    """About how many pixels a sheet for ``provider`` has: :data:`SMALL_SHEET_PIXELS` with a
    context under :data:`FULL_SHEETS` tokens, else :data:`SHEET_PIXELS`."""
    return SMALL_SHEET_PIXELS if provider.context_tokens < FULL_SHEETS else SHEET_PIXELS


def sheet_tokens(provider: Provider) -> int:
    """Image tokens a sheet for ``provider`` counts as: :data:`context.IMAGE_TOKENS` for a
    full-size sheet, in proportion to its pixels for a smaller one."""
    return context.IMAGE_TOKENS * sheet_pixels(provider) // SHEET_PIXELS


# -- consent -----------------------------------------------------------------------------------
def frames_key(provider: Provider) -> str | None:
    """What the user allows once per project before frames go to ``provider``:
    ``frames-v1:<name>@<host[:port]>`` (:func:`cutpoint.assistant.providers.endpoint`); None
    for a provider on this machine (nothing leaves it). Neither ``assistant-v3:`` nor
    ``vision-v2:`` consent covers it."""
    if provider.local:
        return None
    return f"{FRAMES_CONSENT_PREFIX}{provider.name}@{endpoint(provider)}"


def project_key(project: Path) -> str:
    """The project's entry in ``assistant.frames_consent``: its folder, resolved."""
    try:
        return str(project.resolve())
    except OSError:
        return str(project.absolute())


def frames_consent(data: Mapping[str, Any], project: Path) -> tuple[str, ...]:
    """The ``frames-v1:`` keys config ``data`` has for ``project``; none when the section is
    missing or not the shape Cutpoint writes."""
    section = data.get("assistant")
    given = section.get("frames_consent") if isinstance(section, dict) else None
    keys = given.get(project_key(project)) if isinstance(given, dict) else None
    return tuple(k for k in keys if isinstance(k, str)) if isinstance(keys, list) else ()


def frames_view(provider: Provider, project: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    """``{needed, key, host}``: whether the user still has to allow frames of ``project`` to go
    to ``provider`` (``key`` None: a provider on this machine, no consent needed)."""
    key = frames_key(provider)
    return {
        "needed": key is not None and key not in frames_consent(data, project),
        "key": key,
        "host": provider.host,
    }


# -- the sheet ---------------------------------------------------------------------------------
def frame_count(start: float, end: float) -> int:
    """:data:`FEW` frames for a window up to :data:`SHORT_S`, else :data:`MANY`."""
    return FEW if end - start <= SHORT_S else MANY


def frame_times(start: float, end: float) -> list[float]:
    """:func:`frame_count` times spread evenly inside ``start``-``end``, neither end included."""
    return _frame_times(start, end, frame_count(start, end))


def sheet_size(aspect: float, n: int, pixels: int = SHEET_PIXELS) -> tuple[int, int, int, int]:
    """``(tile width, tile height, cols, rows)`` of a sheet of ``n`` frames of ``aspect``
    (width / height): :data:`COLS` per row, the whole sheet about ``pixels`` and no side over
    :data:`MAX_SIDE`."""
    cols = min(COLS, n)
    rows = math.ceil(n / cols)
    sheet_aspect = aspect * cols / rows
    width = math.sqrt(pixels * sheet_aspect)
    height = width / sheet_aspect
    scale = min(1.0, MAX_SIDE / width, MAX_SIDE / height)
    tile_w = max(16, int(width * scale / cols))
    tile_h = max(16, int(height * scale / rows))
    return tile_w, tile_h, cols, rows


def stamp(n: int, t: float) -> str:
    """The text on tile ``n``: its number and source time, ``3  00:12:31.4``."""
    return f"{n}  {clock(t)}"


def grab_frame(src: Path, t: float) -> bytes:
    """One JPEG frame at ``t``, :data:`GRAB_WIDTH` wide, as Vision grabs them."""
    return grab_frame_jpeg(src, t, width=GRAB_WIDTH)


def timeline_sheet(frames: list[bytes], times: list[float], pixels: int = SHEET_PIXELS) -> bytes:
    """The JPEG ``frames`` (grabbed at ``times``) on one JPEG of about ``pixels`` sized by
    :func:`sheet_size` from the first frame's aspect, each stamped (:func:`stamp`) on a dark band at its top left;
    black where the last row is short. ValueError if a frame cannot be decoded."""
    import cv2  # noqa: PLC0415

    tiles = []
    for data in frames:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("a frame could not be decoded")
        tiles.append(img)
    h0, w0 = tiles[0].shape[:2]
    tile_w, tile_h, cols, rows = sheet_size(w0 / h0, len(tiles), pixels)
    scale = max(0.45, tile_h / 480)
    thick = max(1, round(scale * 2))
    out = []
    for i, (img, t) in enumerate(zip(tiles, times, strict=True), 1):
        img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        text = stamp(i, t)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pad = max(3, th // 3)
        cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
        org = (pad, pad + th)
        cv2.putText(
            img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA
        )
        out.append(img)
    out += [np.zeros_like(out[0])] * (rows * cols - len(out))
    grid = np.vstack([np.hstack(out[r * cols : (r + 1) * cols]) for r in range(rows)])
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, SHEET_QUALITY])
    if not ok:
        raise ValueError("the timeline sheet could not be encoded")
    return buf.tobytes()


def sheet_key(src: Path, start: float, end: float, pixels: int = SHEET_PIXELS) -> str:
    """Cache key of a sheet: :data:`SHEET_VERSION`, its size settings (``pixels``: about how
    many it has), the source's path, size and modification time and the window to the
    millisecond. OSError if the source cannot be read."""
    st = src.stat()
    return cache.params_hash(
        {
            "version": SHEET_VERSION,
            "pixels": pixels,
            "side": MAX_SIDE,
            "quality": SHEET_QUALITY,
            "source": str(src),
            "size": st.st_size,
            "mtime": st.st_mtime_ns,
            "window": [round(start, 3), round(end, 3)],
        },
        length=16,
    )


def image_message(text: str, jpeg: bytes, tokens: int = context.IMAGE_TOKENS) -> dict[str, Any]:
    """A user message with ``text`` and the JPEG as the neutral image part every provider
    converts (Anthropic image blocks, OpenAI data URLs, Ollama ``images``). The part's
    ``tokens`` (:func:`sheet_tokens`) is what :func:`cutpoint.assistant.context.size` counts it
    as; no provider sends it on."""
    data = base64.b64encode(jpeg).decode("ascii")
    image = {"type": "image", "media_type": "image/jpeg", "data": data, "tokens": tokens}
    return {"role": "user", "content": [{"type": "text", "text": text}, image]}


def heard(events: list[Any], source: str, start: float, end: float) -> list[str]:
    """The sounds ``events.json`` (``events``) has in ``start``-``end`` of ``source``:
    ``00:12:31.4 Laughter (0.82, 1.4 s)``, at most :data:`MAX_SOUNDS`."""
    out = []
    for e in events:
        if not isinstance(e, dict) or e.get("kind") != "audio" or e.get("source") != source:
            continue
        t0, t1, label = e.get("start"), e.get("end", e.get("start")), e.get("label")
        if not (isinstance(t0, (int, float)) and isinstance(t1, (int, float))):
            continue
        if not isinstance(label, str) or t1 < start or t0 > end:
            continue
        if label == "Explosion" and (e.get("details") or {}).get("confirmed") is not True:
            continue  # an unconfirmed boom is noise (engines, off-screen fire)
        conf = e.get("confidence")
        how = f"{conf:.2f}, " if isinstance(conf, (int, float)) else ""
        out.append(f"{clock(float(t0))} {label} ({how}{float(t1) - float(t0):.1f} s)")
        if len(out) >= MAX_SOUNDS:
            break
    return out


@dataclass
class View:
    """What :meth:`Viewer.view` gives the turn: the result text, the sheet (None: nothing to
    send) and what the chat shows (``event``)."""

    text: str
    sheet: bytes | None
    event: dict[str, Any]


_SAFE = re.compile(r"[^\w.-]")


class Viewer:
    """Builds, caches and gates the sheets of one project for one turn's provider.

    Args:
        project: The project folder (sheets go to ``ui/assistant/sheets``).
        provider: The turn's provider.
        resolve: Where to read a source path from (``source_remap``).
        consent: The ``frames-v1:`` keys the user gave for this project, read at each view, so
            allowing frames during a turn counts at once.
        grab: Grabs one frame as an image file's bytes (:func:`grab_frame`; tests replace it).
    """

    def __init__(
        self,
        project: Path,
        provider: Provider,
        *,
        resolve: Callable[[str], Path] = Path,
        consent: Callable[[], tuple[str, ...]] = tuple,
        grab: Callable[[Path, float], bytes] | None = None,
    ) -> None:
        self.project = project
        self.provider = provider
        self.resolve = resolve
        self.consent = consent
        self.grab = grab or (lambda src, t: grab_frame(src, t))
        self._events: list[Any] | None = None
        self._asked = False  # the turn's consent box is out: later denials show none

    def allowed(self) -> bool:
        key = frames_key(self.provider)
        return key is None or key in self.consent()

    def events(self) -> list[Any]:
        """The project's ``events.json`` entries ([] without one), read once."""
        if self._events is None:
            path = self.project / "events.json"
            data = cache.load_json(path) if _inside(self.project, path) else None
            found = data.get("events") if isinstance(data, dict) else None
            self._events = found if isinstance(found, list) else []
        return self._events

    def _folder(self) -> Path | None:
        folder = self.project / "ui" / "assistant" / SHEETS
        return folder if _inside(self.project, folder) else None

    def sheet(self, real: str, source: str, start: float, end: float) -> tuple[bytes, bool]:
        """``(the sheet of moment real's window, whether it came from the cache)``.

        Raises:
            OSError: The source cannot be read.
            FFmpegError, ValueError: A frame could not be grabbed or decoded.
        """
        src = self.resolve(source)
        pixels = sheet_pixels(self.provider)
        key = sheet_key(src, start, end, pixels)
        name = _SAFE.sub("_", real)[:40]
        folder = self._folder()
        path = folder / f"{name}-{key}.jpg" if folder is not None else None
        if path is not None and path.is_file():
            try:
                return path.read_bytes(), True
            except OSError as err:
                logger.debug("unreadable cached sheet %s: %s", path, err)
        times = frame_times(start, end)
        with ThreadPoolExecutor(min(len(times), GRAB_WORKERS)) as pool:  # #946
            frames = list(pool.map(lambda t: self.grab(src, t), times))
        jpeg = timeline_sheet(frames, times, pixels)
        if path is not None and folder is not None:
            try:
                for old in folder.glob(f"{name}-*.jpg"):  # an older window or source file
                    old.unlink(missing_ok=True)
                cache.save_bytes(path, jpeg, parents=True)
            except OSError as err:  # the sheet still goes out; the next view builds it again
                logger.warning("Could not cache the timeline sheet of %s: %s", real, err)
        return jpeg, False

    def view(
        self, short: str, real: str, item: Mapping[str, Any], row: dict[str, Any], transcript: str
    ) -> View:
        """The view of moment ``real`` (short id ``short``, its window ``item``, its table row
        ``row``): the sheet with its text, or the reason there is none."""
        start, end = float(item["start"]), float(item["end"])
        source = str(item["source"])
        if not self.allowed():
            host = self.provider.host
            text = (
                f"error: the user has not allowed video frames of this project to go to {host}; "
                "nothing was sent. Tell the user they can allow it in the box shown in the chat "
                "and then ask again; meanwhile answer from the text data."
            )
            if self._asked:  # the turn's one box is shown already
                event = {"kind": "tool", "name": NAME, "id": real, "error": "frames not allowed"}
                return View(text, None, event)
            self._asked = True
            event = {
                "kind": "frames_consent",
                "id": real,
                "key": frames_key(self.provider),
                "host": host,
            }
            return View(text, None, event)
        try:
            jpeg, cached = self.sheet(real, source, start, end)
        except (OSError, FFmpegError, ValueError) as err:
            logger.warning("Could not build the timeline sheet of %s: %s", real, err)
            return View(
                f"error: {short}'s video cannot be read, so it cannot be watched",
                None,
                {"kind": "tool", "name": NAME, "id": real, "error": "the video cannot be read"},
            )
        times = frame_times(start, end)
        cols = min(COLS, len(times))
        grid = f"{cols}x{math.ceil(len(times) / cols)}"
        lines = [
            json.dumps(row, ensure_ascii=False),
            f"frames ({grid} grid, reading order, source time): "
            + ", ".join(stamp(i, t).replace("  ", " ") for i, t in enumerate(times, 1)),
        ]
        said = context.transcript_lines(transcript, source, start, end)
        lines += ["transcript:", *(said or ["(no transcript lines)"])]
        sounds = heard(self.events(), source, start, end)
        lines += ["heard:", *(sounds or ["(no detected sounds)"])]
        # the note first: a long result is cut at session.MAX_RESULT_CHARS
        note = f"The timeline sheet of {short} follows as an image.\n"
        text = note + context.data_block("moment view", "\n".join(lines))
        event = {"kind": "tool", "name": NAME, "id": real, "frames": len(times), "cached": cached}
        return View(text, jpeg, event)


def _inside(project: Path, path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(project.resolve())
    except (OSError, RuntimeError):  # RuntimeError: a symlink loop
        return False
