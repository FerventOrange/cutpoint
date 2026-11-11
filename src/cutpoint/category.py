"""Footage category detection: ``--style auto`` on ``analyze`` and ``edit``, and ``cutpoint detect``.

A source is one of five categories (#513):

    action-cam  camera telemetry in the file (a GoPro GPMF track, an Insta360 X3 IMU trailer)
    gameplay    a screen recording: overlay pixels that never change (the HUD), helped by
                screen-recorder marks in the container
    action      a sport or other fast physical activity filmed by a camera that rides along or
                follows the person, without telemetry (only the vision model tells)
    talk        speech covers most of a long recording (known after transcription)
    general     everything else

:func:`probe_source` measures one source (cached per source as :data:`CACHE_FILE`) and
:func:`decide` turns the probes and the speech coverage into a :class:`Detection`, without
I/O. The project gets one category by duration-weighted majority; :func:`resolve_auto` in
:mod:`cutpoint.styles` maps it and the emphasis to a style. Nothing here needs torch: the
frame probe is 32 small grey frames and numpy. A source the frames leave unsure between
gameplay and general also gets the HUD text check (:func:`text_probe`, cached as
:data:`TEXT_FILE`): the HUD OCR engine of the events extra looks for text that stays in place
over a few frames, as a thin game HUD has; without the extra it is skipped. A source still
unsure after that, or one that camera tags alone make general, is shown to the vision model
(:func:`needs_vision`); for a camera-tagged source only an ``action`` answer counts (#958).
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import logging
import math
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.config import remap_source
from cutpoint.errors import FFmpegError, LowMemoryError

logger = logging.getLogger(__name__)

VERSION = 3  # 2: the lower-third box (box, hud_box); 3: per window, bars cropped, size
CACHE_FILE = f"category-v{VERSION}.json"  # in the source's cache folder

ACTION_CAM = "action-cam"
ACTION = "action"
GAMEPLAY = "gameplay"
TALK = "talk"
GENERAL = "general"
CATEGORIES = (ACTION_CAM, ACTION, GAMEPLAY, TALK, GENERAL)
DETECTION_VERSION = 4  # of the detection record (4: action); apart from the probe's VERSION
HYPE, FUNNY = "hype", "funny"  # emphasis: analyze is hype, edit --kind funny-moments funny

# ---- the frame probe
GRID = (320, 180)  # (width, height) of every probe frame
SPREAD = 24  # frames spread over 10-90% of the source, read at READ_SCALE x GRID
READ_SCALE = 2  # spread frames are read this much finer than GRID, cropped, then scaled
WINDOWS = 4  # consecutive groups of the spread frames, each judged on its own (#513)
WINDOW_MIN = 4  # fewer lit frames in a window: not judged
DARK_LUMA = 12.0  # a spread frame darker than this (mean luma: black, a fade) is left out
PAIRS = 4  # pairs of consecutive frames, a native-resolution GRID crop of the centre
PAIR_FRAMES = 4  # frames read per pair, for one that is not a repeat (first_change)
SPAN = (0.1, 0.9)
FRAME_TIMEOUT_S = 60.0
MIN_FRAMES = 6  # fewer spread frames decoded: no frame measures
STATIC_STD = 3.0  # luma levels: a pixel this steady over the spread frames is static
EDGE = 24.0  # luma gradient of the mean frame that counts as an edge
# A HUD's still edges are packed (text, icons, bars); still edges scattered over the picture
# are not one (a repeated pattern such as a grid in the background of several fixed cameras
# lines up between cuts by chance). Only still edges in cells of HUD_CELL x HUD_CELL grid
# pixels with at least HUD_DENSE of them count (#513).
HUD_CELL = 8
HUD_DENSE = 0.2
FLAT = 6.0  # luma gradient below which a pair pixel is flat (noise is measured there)
BAR_LUMA, BAR_STD = 24.0, 3.0  # a letterbox / pillarbox row or column
MARGIN = 2  # pixels left out next to the content's edge
# A lower-third or news-ticker box: a run of rows in the bottom of the picture that moves far
# less than the rest (an opaque or nearly opaque box over the picture) across most of the
# width. Its text stays put like a HUD's, so it counts as neither static HUD nor HUD text.
BOX_FROM = 0.6  # the box lies below this share of the picture's height
BOX_QUIET = 0.35  # a box pixel: luma std at most this share of the picture's median std
BOX_WIDTH = 0.8  # a box row: at least this share of its pixels are box pixels
BOX_ROWS = 0.04  # a box: a run of box rows at least this share of the picture's height

# ---- the decision
# Gameplay score, calibrated on 69 labelled recordings (#513): only a static HUD separates
# gameplay from camera footage. Screen-recorder tags are weak (OBS re-encodes of camera
# footage carry them too); the noise floor and the shake are measured and cached but not
# scored, as fast gameplay shakes and re-encoded camera footage has no sensor noise (checked
# again on 54 labelled files, 2026-11-09: neither separates). A 2:1 frame of 360 video
# counts as camera tags: a re-encode keeps its geometry, not its tags.
BASE_SCORE = 0.3  # no evidence either way
CAMERA_TAGS, SCREEN_TAGS, HUD_WEIGHT = -0.3, 0.1, 0.5
EQUIRECT_MIN_W = 3840  # px: the narrowest 2:1 frame taken as 360 video
HUD_FULL = 0.004  # static HUD share at which the HUD term is full
STILL_SHARE = 0.6  # above this static share the footage is a still or a tripod shot: no HUD
GENERAL_SURE = 0.25  # gameplay score below which general is sure (camera tags)
GAMEPLAY_MIN, GAMEPLAY_SURE = 0.5, 0.6  # gameplay score: ambiguous from, sure from
TALK_SPEECH, TALK_BAND = 0.55, 0.40  # speech coverage: talk from, ambiguous from
TALK_MIN_S = 480.0  # a talk recording is at least this long
SPEECH_BRIDGE_S = 1.0  # pauses up to this long between words count as speech
TELEMETRY_SHARE = 0.5  # action cam wins with telemetry in at least this share of the footage
DISAGREE_SHARE = 0.10  # minorities above this share of the footage: style_sources_disagree
# what can decide a source (SourceDecision.decided_by, Detection.decided_by)
EVIDENCE = ("telemetry", "static HUD", "HUD text", "camera tags", "360 frame", "speech", "vision")

# ---- the HUD text check (OCR), for a source the frame probe leaves unsure
# A thin HUD (a few lines of text) is too small to stay static at GRID; the HUD OCR engine
# finds its text at TEXT_WIDTH. Only text boxes in the same place in most frames count, less
# clocks, dates, speeds and GPS positions (a camera's or dashcam's burned-in overlay) and the
# subtitle band.
TEXT_VERSION = 1
TEXT_FILE = f"category-text-v{TEXT_VERSION}.json"  # in the source's cache folder
TEXT_FRAMES = 8  # frames spread over SPAN
TEXT_WIDTH = 1280  # OCR frame width (a narrower source: its own)
TEXT_MIN_FRAMES = 6  # fewer frames decoded: no check (and nothing cached)
TEXT_KEEP = 0.6  # a text anchor: a box in the same place in this share of the frames
TEXT_MIN_SCORE = 0.7  # recognition score below which an anchor's text does not count (noise)
TEXT_ANCHORS = 3  # kept anchors from which the text is a HUD's
TEXT_WEIGHT = 0.3  # gameplay score of a HUD's text
# gameplay score without the HUD text below which the text alone leaves a source unsure: fixed
# text in a fixed camera's shot (a studio's signs and labels) has anchors as a HUD has (#513)
TEXT_SUPPORT = 0.4
SUBTITLE_BAND = (0.3, 0.7, 0.75)  # x from, x to, y from (share of the frame): subtitles
CLOCK_RE = re.compile(
    r"\d{1,2}\s*[:.]\s*\d{2}(\s*[:.]\s*\d{2})?|\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}|"
    r"\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}|\b[ap]\.?m\b|km/?h|mph|[NSEW]\s?\d{1,3}\.\d|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?\s*\d{1,2}\b|"
    r"\b\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|"
    r"\d{1,3}\s*°\s*\d{1,2}\s*'|\d{1,3}\s*°\s*[CF]\b",
    re.I,
)

_TEXT_FAILED: set[Path] = set()  # cache folders whose text check failed in this process

# ---- the vision question, for a source the rest leaves unsure (#898), or that camera tags
# alone make general (#958: an action camera without telemetry has the tags of a phone clip)
# The shared vision model (local Ollama, or a hosted Vision provider with the user's consent)
# gets a few frames and one generic question. Its answer is cached per source and model.
VISION_VERSION = 3  # 2: the answer action; 3: fast first-person motion is action (#958)
VISION_FILE = f"category-vision-v{VISION_VERSION}.json"  # in the source's cache folder
VISION_FRAMES = 6  # frames spread over SPAN
VISION_WIDTH = 640  # px of each frame sent
VISION_TIMEOUT_S = 120.0  # a cold model load included
V_GAMEPLAY, V_ACTION, V_CAMERA, V_SCREEN, V_OTHER = (
    "gameplay",
    "action",
    "camera",
    "screen",
    "other",
)
VISION_ANSWERS = (V_GAMEPLAY, V_ACTION, V_CAMERA, V_SCREEN, V_OTHER)
VISION_PROMPT = (
    "These are {n} frames spread over one video, in order. What kind of footage is it?\n"
    "gameplay: a recording of a video game being played. Rendered graphics, a game's "
    "interface, menus, maps, loading screens and cutscenes, with or without a webcam, chat "
    "or other stream overlays.\n"
    "action: a sport or other fast physical activity filmed by a camera that rides along "
    "(worn on the body or helmet, mounted on a board) or that follows the person, or a "
    "drone flying fast through the scene (also first-person FPV flight). Fast first-person "
    "motion, where the view rushes past close to the ground, water or obstacles, is action.\n"
    "camera: other real-world scenes filmed with a camera at an everyday pace (a phone, a "
    "vlog, a walk, a dashcam, an event, a slow drone shot), also when edited with captions, "
    "titles or cuts.\n"
    "screen: people talking (a podcast, an interview, a talk to the camera), or a recording "
    "of a computer screen that is not a game (a desktop, apps, a browser, slides).\n"
    "other: none of these, or you cannot tell.\n"
    'Reply with JSON only: {{"footage": "gameplay" | "action" | "camera" | "screen" | "other"}}'
)
VISION_SCHEMA = {
    "type": "object",
    "properties": {"footage": {"type": "string", "enum": list(VISION_ANSWERS)}},
    "required": ["footage"],
}

SCREEN_HANDLERS = ("videohandle",)  # NVIDIA ShadowPlay / the NVIDIA app
CAMERA_RE = re.compile(
    r"gopro|dji|insta360|ambarella|apple|iphone|android|samsung|pixel|sony|canon|nikon|"
    r"panasonic|fujifilm|olympus|osmo|akaso|hero\d",
    re.I,
)
CAMERA_TAG_RE = re.compile(r"^(com\.apple\.quicktime\.|com\.android\.|make$|model$|location)", re.I)


@dataclass
class SourceProbe:
    """What :func:`probe_source` measured of one source.

    Attributes:
        path: The source as given.
        duration: Seconds.
        telemetry: Camera telemetry in the file (:func:`cutpoint.analyze.has_telemetry`).
        handler, encoder: The video stream's ``handler_name`` and ``encoder`` tags, or the
            container's encoder ("" when untagged).
        make, model: The camera's make and model tags ("" when untagged).
        audio_tracks: Audio streams (OBS and editors write several; not scored).
        width, height: The video's display size (0: unknown, a probe cached before it).
        frames: Spread frames decoded (0: no frame measures, such as with telemetry).
        hud_static: Share of the picture that never changes and has edges (a HUD, a logo).
        static_share: Share of the picture that never changes at all.
        box: ``[top, bottom]`` of a lower-third box (:func:`lower_box`), as shares of the
            frame's height, or None.
        hud_box: Share of the picture that is static with edges (as ``hud_static``) inside
            ``box``: the box's own text and edges.
        noise_floor: Luma noise sigma in flat areas of the native-resolution crops.
        shake: Median shift (px) between consecutive frames of the crops.
        hud_text: HUD text anchors the HUD text check kept (None: not checked). Cached
            apart, in :data:`TEXT_FILE`, so not in :meth:`to_json`.
        vision: The vision model's answer (one of :data:`VISION_ANSWERS`; None: not asked).
            Cached apart, in :data:`VISION_FILE`, so not in :meth:`to_json`.
    """

    path: str
    duration: float
    telemetry: bool
    handler: str = ""
    encoder: str = ""
    make: str = ""
    model: str = ""
    audio_tracks: int = 0
    width: int = 0
    height: int = 0
    frames: int = 0
    hud_static: float | None = None
    static_share: float | None = None
    box: list[float] | None = None
    hud_box: float | None = None
    noise_floor: float | None = None
    shake: float | None = None
    hud_text: int | None = None
    vision: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        del data["hud_text"], data["vision"]
        return {"version": VERSION, **data}

    @classmethod
    def from_json(cls, data: Any) -> SourceProbe | None:
        """The probe of a cached record, or None for another version or a broken file."""
        if not isinstance(data, dict) or data.get("version") != VERSION:
            return None
        try:
            return cls(**{k: v for k, v in data.items() if k != "version"})
        except TypeError:
            return None

    @property
    def hud(self) -> float | None:
        """The static HUD share that counts as HUD evidence: ``hud_static`` less the
        lower-third box's (None without frame measures)."""
        if self.hud_static is None:
            return None
        return max(0.0, self.hud_static - (self.hud_box or 0.0))

    @property
    def camera_marks(self) -> bool:
        """The container names a camera (make, model, a camera maker's handler)."""
        return bool(self.make or self.model) or bool(
            CAMERA_RE.search(f"{self.handler} {self.encoder}")
        )

    @property
    def screen_marks(self) -> bool:
        """The container carries a screen recorder's own signature (ShadowPlay's handler) and
        no camera marks. Several audio tracks are no sign: editors export them too."""
        return not self.camera_marks and self.handler.lower() in SCREEN_HANDLERS

    @property
    def equirect(self) -> bool:
        """The picture is a 2:1 frame of at least :data:`EQUIRECT_MIN_W` pixels: a stitched
        360 video (a 2:1 phone screen recording is narrower)."""
        return self.width >= EQUIRECT_MIN_W and abs(self.width - 2 * self.height) <= 2


# --------------------------------------------------------------------------- probing
def _tags(stream_or_format: Mapping[str, Any]) -> dict[str, str]:
    tags = stream_or_format.get("tags")
    return {str(k): str(v) for k, v in tags.items()} if isinstance(tags, dict) else {}


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    """Handler, encoder, make, model and the audio track count of ``ffprobe`` output."""
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    vtags, ftags = _tags(video), _tags(data.get("format") or {})
    make = model = ""
    for key, value in {**ftags, **vtags}.items():
        low = key.lower()
        if low.endswith(("make", "manufacturer")):
            make = make or value
        elif low.endswith("model"):
            model = model or value
        elif CAMERA_TAG_RE.match(key) and not make:
            make = key.split(".")[1] if low.startswith("com.") else "camera"
    return {
        "handler": vtags.get("handler_name", ""),
        "encoder": vtags.get("encoder") or ftags.get("encoder", ""),
        "make": make,
        "model": model,
        "audio_tracks": sum(s.get("codec_type") == "audio" for s in streams),
    }


def _grab(
    io: Path, t: float, vf: str, count: int, size: tuple[int, int] = GRID
) -> list[np.ndarray]:
    """Up to ``count`` consecutive grey ``size`` frames of ``io`` from ``t`` ([] on a
    failure)."""
    from cutpoint.ffmpeg_utils import decode_raw  # noqa: PLC0415

    out: list[np.ndarray] = []
    try:
        for _t, frame in decode_raw(
            io,
            size=size,
            vf=vf,
            pix_fmt="gray",
            start=max(0.0, t),
            duration=2.0,
            hwaccel=None,
            timeout=FRAME_TIMEOUT_S,
        ):
            out.append(frame.astype(np.float32))
            if len(out) >= count:
                break
    except FFmpegError as err:
        logger.debug("frame probe of %s at %.1fs failed: %s", io, t, err)
    return out


def _content_box(mean: np.ndarray, std: np.ndarray) -> tuple[slice, slice]:
    """The picture inside letterbox and pillarbox bars, less :data:`MARGIN`."""

    def inner(m: np.ndarray, s: np.ndarray) -> tuple[int, int]:
        bar = (m < BAR_LUMA) & (s < BAR_STD)
        lo, hi = 0, len(bar)
        while lo < hi and bar[lo]:
            lo += 1
        while hi > lo and bar[hi - 1]:
            hi -= 1
        return lo + MARGIN, max(lo + MARGIN + 1, hi - MARGIN)

    r0, r1 = inner(mean.mean(axis=1), std.mean(axis=1))
    c0, c1 = inner(mean.mean(axis=0), std.mean(axis=0))
    return slice(r0, r1), slice(c0, c1)


def _gradient(img: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude (|dx| + |dy|), same shape, edges zero."""
    g = np.zeros_like(img)
    g[1:-1, 1:-1] = np.abs(img[1:-1, 2:] - img[1:-1, :-2]) + np.abs(img[2:, 1:-1] - img[:-2, 1:-1])
    return g


def lower_box(std: np.ndarray) -> tuple[int, int] | None:
    """Rows ``(top, bottom)`` (bottom exclusive) of a lower-third box in the per-pixel luma
    std of the picture (bars left out): the longest run, below :data:`BOX_FROM` of its
    height, of rows that are mostly (:data:`BOX_WIDTH`) far quieter (:data:`BOX_QUIET`)
    than the picture's median; None when no run is :data:`BOX_ROWS` of the height."""
    h = std.shape[0]
    quiet = std < max(STATIC_STD, BOX_QUIET * float(np.median(std)))
    rows = quiet.mean(axis=1) >= BOX_WIDTH
    best, start = (0, 0), None
    for y in range(int(h * BOX_FROM), h + 1):
        if y < h and rows[y]:
            start = y if start is None else start
            continue
        if start is not None and y - start > best[1] - best[0]:
            best = (start, y)
        start = None
    return best if best[1] - best[0] >= max(1, round(BOX_ROWS * h)) else None


def _packed(mask: np.ndarray) -> np.ndarray:
    """``mask`` with only its pixels in cells (:data:`HUD_CELL`) where at least
    :data:`HUD_DENSE` of the pixels are set; a partial cell at the right or bottom edge is
    judged on its own pixels."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    for y in range(0, h, HUD_CELL):
        for x in range(0, w, HUD_CELL):
            cell = mask[y : y + HUD_CELL, x : x + HUD_CELL]
            if cell.mean() >= HUD_DENSE:
                out[y : y + HUD_CELL, x : x + HUD_CELL] = cell
    return out


def static_measures(frames: np.ndarray) -> dict[str, Any]:
    """``hud_static``, ``static_share`` and, with a lower-third box (:func:`lower_box`),
    ``box`` and ``hud_box`` of spread frames ``(n, h, w)``: the share of the picture (bars
    left out) that stays still and has packed edges (:func:`_packed`), that stays still at
    all, the box's rows as shares of the frame's height, and the still, packed edges' share
    of the picture inside it."""
    mean, std = frames.mean(axis=0), frames.std(axis=0)
    rows, cols = _content_box(mean, std)
    static = std[rows, cols] < STATIC_STD
    edgy = _gradient(mean)[rows, cols] > EDGE
    if not static.size:
        return {"hud_static": 0.0, "static_share": 0.0}
    hud = _packed(static & edgy)
    out: dict[str, Any] = {"hud_static": float(hud.mean()), "static_share": float(static.mean())}
    box = lower_box(std[rows, cols])
    if box is not None:
        out["box"] = [round((rows.start + y) / frames.shape[1], 4) for y in box]
        out["hud_box"] = float(hud[box[0] : box[1]].sum() / hud.size)
    return out


def noise_sigma(img: np.ndarray) -> float | None:
    """Immerkaer's noise estimate (luma sigma) over the flat pixels of ``img``; None when
    too little of it is flat."""
    lap = (
        img[:-2, :-2] - 2 * img[:-2, 1:-1] + img[:-2, 2:]
        - 2 * img[1:-1, :-2] + 4 * img[1:-1, 1:-1] - 2 * img[1:-1, 2:]
        + img[2:, :-2] - 2 * img[2:, 1:-1] + img[2:, 2:]
    )  # fmt: skip
    flat = _gradient(img)[1:-1, 1:-1] < FLAT
    if flat.mean() < 0.05:
        return None
    return float(math.sqrt(math.pi / 2) * np.abs(lap[flat]).mean() / 6)


def shift(a: np.ndarray, b: np.ndarray) -> float:
    """Magnitude (px) of the global shift between frames ``a`` and ``b`` (phase correlation)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-9)).real
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = dy - a.shape[0] if dy > a.shape[0] // 2 else dy
    dx = dx - a.shape[1] if dx > a.shape[1] // 2 else dx
    return float(math.hypot(dx, dy))


def first_change(frames: list[np.ndarray]) -> list[np.ndarray]:
    """The first two consecutive ``frames`` that differ (a 60 fps recording of 30 fps
    footage repeats each frame), else the first two."""
    for a, b in itertools.pairwise(frames):
        if float(np.abs(a - b).mean()) > 0.5:
            return [a, b]
    return frames[:2]


def _bars(frames: np.ndarray) -> tuple[slice, slice]:
    """The picture inside black bars that stay over ``frames`` ``(n, h, w)``: rows and
    columns whose every pixel is dark (:data:`BAR_LUMA`) and steady (:data:`BAR_STD`), as a
    letterbox, a pillarbox or a game window inside a larger canvas has."""
    mean, std = frames.mean(axis=0), frames.std(axis=0)
    bar = (mean < BAR_LUMA) & (std < BAR_STD)

    def inner(lines: np.ndarray) -> tuple[int, int]:
        lo, hi = 0, len(lines)
        while lo < hi and lines[lo]:
            lo += 1
        while hi > lo and lines[hi - 1]:
            hi -= 1
        return lo, hi

    r0, r1 = inner(bar.all(axis=1))
    c0, c1 = inner(bar.all(axis=0))
    return slice(r0, r1), slice(c0, c1)


def _to_grid(frames: np.ndarray) -> np.ndarray:
    """``frames`` ``(n, h, w)`` cropped to the picture inside their black bars (:func:`_bars`)
    and scaled to :data:`GRID`."""
    import cv2  # noqa: PLC0415

    rows, cols = _bars(frames)
    if rows.stop - rows.start < GRID[1] // 4 or cols.stop - cols.start < GRID[0] // 4:
        rows, cols = slice(None), slice(None)  # all dark: nothing to crop to
    return np.stack(
        [
            cv2.resize(f[rows, cols], GRID, interpolation=cv2.INTER_AREA).astype(np.float32)
            for f in frames
        ]
    )


def window_measures(spread: list[np.ndarray]) -> dict[str, Any]:
    """:func:`static_measures` of the spread frames (in time order) judged per window: the
    frames in :data:`WINDOWS` consecutive groups, dark frames left out (:data:`DARK_LUMA`), each
    group cropped to the picture inside its black bars and scaled to :data:`GRID`. The HUD
    measures (``hud_static``, ``box``, ``hud_box``) are those of the window with the most HUD
    evidence that is no still picture, so menus, loading, dark scenes and cutscenes in other
    windows do not dilute it; ``static_share`` is that of all the lit frames (a still picture
    overall)."""
    lit = [f for f in spread if float(f.mean()) >= DARK_LUMA]
    pooled = static_measures(_to_grid(np.stack(lit if len(lit) >= WINDOW_MIN else spread)))
    size = math.ceil(len(spread) / WINDOWS)
    best: dict[str, Any] | None = None
    for i in range(0, len(spread), size):
        part = [f for f in spread[i : i + size] if float(f.mean()) >= DARK_LUMA]
        if len(part) < WINDOW_MIN:
            continue
        m = static_measures(_to_grid(np.stack(part)))
        if m["static_share"] > STILL_SHARE:
            continue
        hud = m["hud_static"] - m.get("hud_box", 0.0)
        if best is None or hud > best["hud_static"] - best.get("hud_box", 0.0):
            best = m
    if best is None:  # every window a still or dark: the frames as a whole
        return pooled
    return {**best, "static_share": pooled["static_share"]}


def frame_measures(spread: list[np.ndarray], pairs: list[list[np.ndarray]]) -> dict[str, Any]:
    """The frame fields of :class:`SourceProbe` from decoded frames (``spread`` at any size:
    :func:`window_measures` scales them to :data:`GRID`)."""
    out: dict[str, Any] = {"frames": len(spread)}
    if len(spread) >= MIN_FRAMES:
        out.update(window_measures(spread))
    full = [p for p in pairs if len(p) == 2]
    noise = [s for p in full for f in p if (s := noise_sigma(f)) is not None]
    if noise:
        out["noise_floor"] = float(np.median(noise))
    moving = [shift(a, b) for a, b in full if float(np.abs(a - b).mean()) > 0.5]
    if full:  # duplicated frames (a still, a paused game) move 0 px
        out["shake"] = float(np.median(moving)) if moving else 0.0
    return out


def _frame_probe(io: Path, duration: float, width: int, height: int) -> dict[str, Any]:
    lo, hi = SPAN
    times = [duration * (lo + (hi - lo) * i / (SPREAD - 1)) for i in range(SPREAD)]
    read = (GRID[0] * READ_SCALE, GRID[1] * READ_SCALE)  # finer: bars are cropped off later
    spread = [f[0] for t in times if (f := _grab(io, t, f"scale={read[0]}:{read[1]}", 1, read))]
    scale = f"scale={GRID[0]}:{GRID[1]}"
    crop = f"crop={GRID[0]}:{GRID[1]}" if width >= GRID[0] and height >= GRID[1] else scale
    pair_t = [duration * (lo + (hi - lo) * (i + 0.5) / PAIRS) for i in range(PAIRS)]
    pairs = [first_change(_grab(io, t, crop, PAIR_FRAMES)) for t in pair_t]
    return frame_measures(spread, pairs)


# --------------------------------------------------------------------------- HUD text
Box = tuple[float, float, float, float]  # (x0, y0, x1, y1) in frame pixels


def _same_place(a: Box, b: Box) -> bool:
    """Two text boxes of one HUD element: the same line (centres within half the lower
    box's height, heights within 1.6x) and overlapping by half the narrower box, so a
    number whose width changes still matches."""
    ah, bh = a[3] - a[1], b[3] - b[1]
    if min(ah, bh) <= 0 or max(ah, bh) > 1.6 * min(ah, bh):
        return False
    if abs((a[1] + a[3]) - (b[1] + b[3])) / 2 > min(ah, bh) / 2:
        return False
    return min(a[2], b[2]) - max(a[0], b[0]) >= 0.5 * min(a[2] - a[0], b[2] - b[0])


def text_anchors(frames: Sequence[Sequence[Box]]) -> list[tuple[int, Box]]:
    """``(first frame, box)`` of every text anchor of the text boxes per frame: a box in
    the same place (:func:`_same_place`) in at least :data:`TEXT_KEEP` of the frames."""
    groups: list[tuple[int, Box, set[int]]] = []
    for i, boxes in enumerate(frames):
        for b in boxes:
            for _first, rep, seen in groups:
                if i not in seen and _same_place(rep, b):
                    seen.add(i)
                    break
            else:
                groups.append((i, b, {i}))
    need = max(2, math.ceil(TEXT_KEEP * len(frames)))
    return [(first, rep) for first, rep, seen in groups if len(seen) >= need]


def hud_anchors(anchors: Sequence[Mapping[str, Any]], box: Sequence[float] | None = None) -> int:
    """How many of the HUD text check's ``anchors`` (``box`` as shares of the frame,
    ``text``, ``score``) are a HUD's: read (score of at least :data:`TEXT_MIN_SCORE`), not
    a clock, date, speed or position (a camera's burned-in overlay), not in the subtitle
    band and not in the lower-third ``box`` (:attr:`SourceProbe.box`)."""
    x_lo, x_hi, y_lo = SUBTITLE_BAND
    kept = 0
    for a in anchors:
        x0, y0, x1, y1 = a["box"]
        text = str(a.get("text") or "")
        if a.get("score", 0.0) < TEXT_MIN_SCORE or not text.strip() or CLOCK_RE.search(text):
            continue
        if (y0 + y1) / 2 >= y_lo and x_lo <= (x0 + x1) / 2 <= x_hi:
            continue
        if box is not None and box[0] <= (y0 + y1) / 2 <= box[1]:
            continue
        kept += 1
    return kept


def _text_frames(io: Path, duration: float, width: int, height: int) -> list[np.ndarray]:
    """:data:`TEXT_FRAMES` BGR frames of ``io`` spread over :data:`SPAN`, at most
    :data:`TEXT_WIDTH` wide (a failed read is left out)."""
    from cutpoint.ffmpeg_utils import decode_raw  # noqa: PLC0415

    w = min(TEXT_WIDTH, width) // 2 * 2
    size = (w, max(2, round(w * height / width / 2) * 2))
    lo, hi = SPAN
    out = []
    for i in range(TEXT_FRAMES):
        t = duration * (lo + (hi - lo) * i / (TEXT_FRAMES - 1))
        try:
            for _t, frame in decode_raw(
                io,
                size=size,
                vf=f"scale={size[0]}:{size[1]}",
                pix_fmt="bgr24",
                start=t,
                duration=2.0,
                hwaccel=None,
                timeout=FRAME_TIMEOUT_S,
            ):
                out.append(frame)
                break
        except FFmpegError as err:
            logger.debug("text probe of %s at %.1fs failed: %s", io, t, err)
    return out


def text_probe(io: Path, duration: float, width: int, height: int) -> dict[str, Any] | None:
    """The HUD text check of one source, as cached in :data:`TEXT_FILE`: text detection
    (no recognition) on :data:`TEXT_FRAMES` frames, then one recognition of each text
    anchor (:func:`text_anchors`) in the first frame that has it. None when too few
    frames were read.

    Needs ``rapidocr_onnxruntime`` (the events extra).
    """
    from cutpoint.events.killfeed import OcrEngine  # noqa: PLC0415
    from cutpoint.events.ocr_raw import engine_version  # noqa: PLC0415

    frames = _text_frames(io, duration, width, height)
    if len(frames) < TEXT_MIN_FRAMES:
        return None
    engine = OcrEngine()

    def boxes(frame: np.ndarray) -> list[Box]:
        found = []
        for quad in engine.detect(frame):
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            found.append((min(xs), min(ys), max(xs), max(ys)))
        return found

    anchors = text_anchors(engine.map(boxes, frames))
    fh, fw = frames[0].shape[:2]
    out = []
    for first, (x0, y0, x1, y1) in anchors:
        read = engine.read_rows(frames[first], [(y0, x0, y1, x1)], 0.0)
        text = " ".join(t for _box, t, _s in read)
        score = min((s for _box, _t, s in read), default=0.0)
        out.append(
            {
                "box": [round(x0 / fw, 4), round(y0 / fh, 4), round(x1 / fw, 4), round(y1 / fh, 4)],
                "text": text,
                "score": round(score, 3),
            }
        )
    return {
        "version": TEXT_VERSION,
        "engine": engine_version(),
        "frames": len(frames),
        "anchors": out,
    }


def ocr_installed() -> bool:
    """Whether the HUD OCR engine (the events extra, not in Cutpoint lite) is installed."""
    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


def needs_text(p: SourceProbe) -> bool:
    """Whether the HUD text check can settle ``p``: no telemetry, no camera tags, a frame
    probe that is not a still picture, and a gameplay score that is neither sure gameplay
    nor sure general (the text never makes camera-tagged footage or a still gameplay)."""
    if p.telemetry or p.camera_marks or p.hud_static is None:
        return False
    if p.static_share is not None and p.static_share > STILL_SHARE:
        return False
    return GENERAL_SURE <= gameplay_score(replace(p, hud_text=None))[0] < GAMEPLAY_SURE


def _with_text(found: SourceProbe, cdir: Path, io: Path | None = None) -> SourceProbe:
    """``found`` with :attr:`SourceProbe.hud_text` when :func:`needs_text`: from
    :data:`TEXT_FILE`, else from :func:`text_probe` (cached); unchanged without the OCR
    engine (Cutpoint lite) or when the check fails (tried once per process)."""
    if not needs_text(found):
        return found
    from cutpoint.events.ocr_raw import engine_version  # noqa: PLC0415

    rec = cache.load_json(cdir / TEXT_FILE)
    fresh = (
        isinstance(rec, dict)
        and rec.get("version") == TEXT_VERSION
        and rec.get("engine") == engine_version()
        and isinstance(rec.get("anchors"), list)
    )
    if not fresh:
        if cdir in _TEXT_FAILED:
            return found
        if not ocr_installed():
            logger.debug("HUD text check of %s skipped: no OCR engine", found.path)
            return found
        from cutpoint.ffmpeg_utils import probe  # noqa: PLC0415

        io = io or remap_source(Path(found.path))
        t0 = time.perf_counter()
        try:
            info = probe(io)
            rec = text_probe(io, info.duration, info.width, info.height)
        except LowMemoryError:  # the memory guard stops the run; no guess without the HUD check
            raise
        except Exception as err:  # noqa: BLE001 - an OCR or decode failure: the frame probe stands
            logger.warning("HUD text check of %s failed: %s", Path(found.path).name, err)
            _TEXT_FAILED.add(cdir)
            return found
        if rec is None:
            _TEXT_FAILED.add(cdir)
            return found
        logger.debug("HUD text check of %s: %.1fs", found.path, time.perf_counter() - t0)
        cache.save_json(cdir / TEXT_FILE, rec)
    return replace(found, hud_text=hud_anchors(rec["anchors"], found.box))


# --------------------------------------------------------------------------- vision (#898)
def vision_frames(io: Path, duration: float) -> list[bytes]:
    """:data:`VISION_FRAMES` JPEG frames of ``io`` spread over :data:`SPAN`,
    :data:`VISION_WIDTH` wide (a failed grab is left out)."""
    from cutpoint.ffmpeg_utils import grab_frame_jpeg  # noqa: PLC0415

    lo, hi = SPAN
    out = []
    for i in range(VISION_FRAMES):
        t = duration * (lo + (hi - lo) * i / (VISION_FRAMES - 1))
        try:
            out.append(grab_frame_jpeg(io, t, VISION_WIDTH))
        except FFmpegError as err:
            logger.debug("vision frame of %s at %.1fs failed: %s", io, t, err)
    return out


class FootageVision:
    """The vision question for the sources of one detection that :func:`needs_vision`: one
    client of the shared vision model (:class:`cutpoint.vision.VisionRater`: the local Ollama,
    or the hosted Vision provider the user agreed to), checked once and freed by :meth:`close`.

    ``ollama_url`` and ``ollama_model`` are the run's ``--ollama-url`` and ``--ollama-model``
    (they pick the Ollama over a hosted Vision role, as for the rating). ``explicit``: the
    user asked for vision (``--vision``, ``"vision": true``).

    Quiet: in Cutpoint lite, when no model answers or memory is short, one INFO line, and the
    sources stay as the rest decided them. Without the user's Vision consent the same, unless
    ``explicit``: then the ``vision_consent_needed`` WARNING (once per run).
    """

    def __init__(
        self,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        explicit: bool = False,
    ) -> None:
        self._url, self._model, self._explicit = ollama_url, ollama_model, explicit
        self._client: Any = None
        self._made = False
        self._ready: bool | None = None  # None: not checked yet

    def _skip(self, why: str) -> None:
        self._ready = False
        logger.info("Footage category: the vision model is not asked (%s)", why)

    def client(self) -> Any:
        """The :class:`~cutpoint.vision.VisionRater`, not checked yet (None in Cutpoint lite
        or with a bad Ollama URL; the reason is logged once)."""
        if not self._made:
            self._made = True
            from cutpoint.extras import edition  # noqa: PLC0415
            from cutpoint.vision import VisionRater  # noqa: PLC0415

            if edition() == "lite":
                self._skip("not in Cutpoint lite")
                return None
            try:
                self._client = VisionRater(self._url, self._model)
            except ValueError as err:  # a bad port in OLLAMA_HOST / config.json
                self._skip(str(err))
        return self._client

    def key(self) -> str | None:
        """The cache key of the model's answers, ``<model>@<host>`` (None without a client)."""
        rater = self.client()
        return None if rater is None else f"{rater.model}@{rater.host}"

    def _rater(self) -> Any:
        """The client once it is checked (consent, then the model answering), else None."""
        rater = self.client()
        if self._ready is None and rater is not None:
            if rater.consent_needed:
                if self._explicit:
                    rater.warn_consent()
                    self._ready = False
                else:
                    self._skip(
                        "the Vision provider needs your consent in the Models tab of cutpoint ui"
                    )
            else:
                self._ready = rater.available()  # it logs why not (INFO)
        return rater if self._ready else None

    def ready(self) -> bool:
        """Whether the model can be asked: the client is checked (consent, answering) and its
        model id is known (:meth:`key`)."""
        return self._rater() is not None

    def ask(self, io: Path, duration: float) -> str | None:
        """The model's answer (one of :data:`VISION_ANSWERS`) about :func:`vision_frames` of
        ``io``; None when it is not asked, fails or answers something else."""
        rater = self._rater()
        if rater is None:
            return None
        frames = vision_frames(io, duration)
        if len(frames) < 2:
            return None
        try:
            reply = rater.ask_json(
                VISION_PROMPT.format(n=len(frames)),
                frames,
                VISION_SCHEMA,
                num_predict=256,
                timeout=VISION_TIMEOUT_S,
            )
        except LowMemoryError as err:
            self._skip(str(err))
            return None
        if reply is None:  # the request failed (logged): no more requests in this detection
            self._ready = False
            return None
        answer = reply.get("footage")
        return answer if answer in VISION_ANSWERS else None

    def close(self) -> None:
        """Free the model's memory for what runs next (only if a request went to it)."""
        if self._client is not None:
            self._client.unload()


@contextlib.contextmanager
def footage_vision(
    vision: bool | None = None, ollama_url: str | None = None, ollama_model: str | None = None
) -> Iterator[FootageVision | None]:
    """A :class:`FootageVision` for one detection, closed at the end. ``vision``: True = asked
    for (a missing consent is a WARNING), None = the default (asked quietly), False = off (None
    is yielded). ``ollama_url``, ``ollama_model``: see :class:`FootageVision`."""
    vision = None if vision is False else FootageVision(ollama_url, ollama_model, vision is True)
    try:
        yield vision
    finally:
        if vision is not None:
            vision.close()


def needs_vision(p: SourceProbe) -> bool:
    """Whether to ask the vision model about ``p``: no telemetry, a frame probe, and the rest
    (the HUD text check included) leaves it unsure, or camera tags alone make it sure general
    (an action camera without telemetry has the tags of a phone clip; only an ``action``
    answer counts for it, see :func:`_source`). A 360 source (a 2:1 frame) is never asked,
    whatever its tags."""
    if p.telemetry or p.frames < MIN_FRAMES:
        return False
    found = _source(replace(p, vision=None), None)
    return found.ambiguous or (found.decided_by == "camera tags" and not p.equirect)


def _with_vision(
    found: SourceProbe, cdir: Path, vision: FootageVision | None, io: Path | None = None
) -> SourceProbe:
    """``found`` with :attr:`SourceProbe.vision` when :func:`needs_vision`: from
    :data:`VISION_FILE` (per model), else asked and cached; unchanged without ``vision`` or
    an answer."""
    if vision is None or not needs_vision(found):
        return found
    rec = cache.load_json(cdir / VISION_FILE)
    answers = rec.get("answers") if isinstance(rec, dict) else None
    if not isinstance(answers, dict) or rec.get("version") != VISION_VERSION:
        answers = {}
    key = vision.key()
    if key is None:
        return found
    if answers.get(key) not in VISION_ANSWERS:
        if vision.ready():  # a hosted model's id is known once it is checked: look again
            key = vision.key() or key
        if answers.get(key) in VISION_ANSWERS:
            return replace(found, vision=answers[key])
        t0 = time.perf_counter()
        answer = vision.ask(io or remap_source(Path(found.path)), found.duration)
        if answer is None:
            return found
        logger.info(
            "The vision model sees %s in the footage of %s (%.1fs)",
            answer,
            Path(found.path).name,
            time.perf_counter() - t0,
        )
        answers = {**answers, key: answer}
        cache.save_json(cdir / VISION_FILE, {"version": VISION_VERSION, "answers": answers})
    return replace(found, vision=answers[key])


def probe_source(src: Path, vision: FootageVision | None = None) -> SourceProbe:
    """Measure ``src`` for :func:`decide`; cached in its cache folder as :data:`CACHE_FILE`.

    The telemetry check and the container tags are always read; the frame probe
    (:data:`SPREAD` scaled frames and :data:`PAIRS` pairs of crops) only without telemetry,
    the HUD text check only when the rest leaves the source unsure (:func:`needs_text`), and
    with ``vision`` the vision model only when it is still unsure or camera tags alone make
    it general (:func:`needs_vision`).

    Raises:
        FFmpegError: If ffprobe cannot read the file.
    """
    from cutpoint.analyze import has_telemetry  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import ffprobe_json, probe  # noqa: PLC0415

    cdir = cache.source_dir(src)
    cached = SourceProbe.from_json(cache.load_json(cdir / CACHE_FILE))
    if cached is not None:  # saved under the path of its first run, maybe another remap
        return _with_vision(_with_text(replace(cached, path=str(src)), cdir), cdir, vision)
    io = remap_source(src)
    info = probe(io)
    meta = _metadata(ffprobe_json(io, show_format=True, show_streams=True, timeout=60))
    found = SourceProbe(
        str(src),
        round(info.duration, 3),
        has_telemetry(src),
        **meta,
        width=info.width,
        height=info.height,
    )
    if not found.telemetry:
        for key, value in _frame_probe(io, info.duration, info.width, info.height).items():
            setattr(found, key, round(value, 5) if isinstance(value, float) else value)
    if found.telemetry or found.frames >= MIN_FRAMES:  # else a read failure that may pass
        cache.save_json(cdir / CACHE_FILE, found.to_json())
    return _with_vision(_with_text(found, cdir, io), cdir, vision, io)


# --------------------------------------------------------------------------- speech
def speech_coverage(segments: Sequence[Mapping[str, Any]], duration: float) -> float:
    """Share of ``duration`` covered by spoken words (pauses up to :data:`SPEECH_BRIDGE_S`
    between words count), 0-1."""
    spans = sorted(
        (float(w["start"]), float(w["end"])) for s in segments for w in s.get("words") or []
    )
    total, cur = 0.0, None
    for a, b in spans:
        if cur is not None and a - cur[1] <= SPEECH_BRIDGE_S:
            cur[1] = max(cur[1], b)
            continue
        if cur is not None:
            total += cur[1] - cur[0]
        cur = [a, b]
    if cur is not None:
        total += cur[1] - cur[0]
    return min(1.0, total / duration) if duration > 0 else 0.0


# --------------------------------------------------------------------------- the decision
@dataclass
class SourceDecision:
    """The category of one source and why."""

    path: str
    duration: float
    category: str
    ambiguous: bool
    reason: str
    gameplay: float | None = None  # gameplay score 0-1 (None with telemetry)
    speech: float | None = None  # speech coverage 0-1 (None: not transcribed yet)
    # the evidence that decided (:data:`EVIDENCE`); None when ambiguous
    decided_by: str | None = None


@dataclass
class Detection:
    """The project's category (:func:`decide`).

    Attributes:
        category: One of :data:`CATEGORIES`.
        ambiguous: The decision is not sure: the caller falls back to its default style.
        reason: Short text for the log line, such as ``screen recorder tags, static HUD 0.9%``.
        scores: Share of the footage's duration per category.
        sources: Per-source decisions, in input order.
        pending: The talk check needs the speech coverage of a general or action source of at least
            :data:`TALK_MIN_S` (not given yet); ``category`` is general or action until
            :func:`decide` gets it.
        disagree: Sources of another category than the project's, when they are more than
            :data:`DISAGREE_SHARE` of the footage (else empty).
        suggest: A style to suggest (``commentary`` for gameplay that is mostly speech),
            or None.
        decided_by: The evidence (:data:`EVIDENCE`) that decided the sure sources of the
            category, most footage first; empty when the detection is ambiguous.
    """

    category: str
    ambiguous: bool
    reason: str
    scores: dict[str, float]
    sources: list[SourceDecision]
    pending: bool = False
    disagree: list[str] = field(default_factory=list)
    suggest: str | None = None
    decided_by: list[str] = field(default_factory=list)

    def record(self, emphasis: str, style: str, fallback: str | None) -> dict[str, Any]:
        """The ``detection`` block of highlights.json and of ``--json``."""
        return {
            "version": DETECTION_VERSION,
            "category": self.category,
            "emphasis": emphasis,
            "style": style,
            "reason": self.reason,
            "ambiguous": self.ambiguous,
            "scores": self.scores,
            "fallback": fallback,
            "suggest": self.suggest,
            "decided_by": self.decided_by,
            "sources": [asdict(s) for s in self.sources],
        }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%" if x < 0.1 else f"{100 * x:.0f}%"


def gameplay_score(p: SourceProbe) -> tuple[float, list[str]]:
    """Gameplay score (0-1) of a probe without telemetry, and the evidence behind it."""
    score, why = BASE_SCORE, []
    if p.camera_marks:
        score += CAMERA_TAGS
        why.append(f"camera tags ({p.make or p.handler or p.encoder})".replace("  ", " "))
    elif p.equirect:
        score += CAMERA_TAGS
        why.append(f"360 frame ({p.width}x{p.height})")
    elif p.screen_marks:
        score += SCREEN_TAGS
        why.append("screen recorder tags")
    if p.hud is not None:
        if p.static_share is not None and p.static_share > STILL_SHARE:
            why.append(f"still picture ({_pct(p.static_share)} static)")
        else:
            hud = min(1.0, p.hud / HUD_FULL)
            score += HUD_WEIGHT * hud
            why.append(f"static HUD {_pct(p.hud)}" if hud >= 0.5 else "no static HUD")
            if p.box is not None:  # its text stays put like a HUD's: counted as neither
                why.append("lower-third box")
    if p.hud_text is not None:  # only checked when the above is unsure (needs_text)
        if p.hud_text >= TEXT_ANCHORS:
            score += TEXT_WEIGHT
            why.append(f"HUD text in {p.hud_text} places")
        else:
            why.append("no HUD text")
    return round(max(0.0, min(1.0, score)), 3), why


def _text_alone(p: SourceProbe) -> bool:
    """``p``'s gameplay score rests on its HUD text alone (:data:`TEXT_SUPPORT`)."""
    lifted = (p.hud_text or 0) >= TEXT_ANCHORS
    return lifted and gameplay_score(replace(p, hud_text=None))[0] < TEXT_SUPPORT


def _gameplay_evidence(p: SourceProbe) -> str:
    """What made ``p`` sure gameplay without the vision model: ``HUD text`` when the source
    is sure only with it, else ``static HUD``."""
    lifted = (p.hud_text or 0) >= TEXT_ANCHORS
    if lifted and gameplay_score(replace(p, hud_text=None))[0] < GAMEPLAY_SURE:
        return "HUD text"
    return "static HUD"


def _general_evidence(p: SourceProbe) -> str:
    """What made ``p`` sure general: ``camera tags`` or ``360 frame``."""
    return "camera tags" if p.camera_marks else "360 frame"


def _source(p: SourceProbe, speech: float | None) -> SourceDecision:
    """The category of one source. The vision model's answer (:attr:`SourceProbe.vision`,
    only asked when the rest is unsure or camera tags alone decide, :func:`needs_vision`)
    settles gameplay or not; ``other`` leaves it open. A source the camera tags make sure
    general without it stays so unless the answer is ``action`` (re-exports and phone clips
    carry the tags too, #958)."""
    if p.telemetry:
        return SourceDecision(
            p.path, p.duration, ACTION_CAM, False, "camera telemetry", None, speech, "telemetry"
        )
    if p.vision is not None and p.camera_marks:
        # asked only because the tags decide (needs_vision): the answer can make it action,
        # nothing else; the speech (talk) is weighed as without it
        plain = _source(replace(p, vision=None), speech)
        by_tags = _source(replace(p, vision=None), None).decided_by == "camera tags"
        action = p.vision == V_ACTION and plain.decided_by == "camera tags" and not p.equirect
        if by_tags and not action:
            return plain
    score, why = gameplay_score(p)
    seen = p.vision if p.vision in (V_GAMEPLAY, V_ACTION, V_CAMERA, V_SCREEN) else None
    if p.vision in VISION_ANSWERS:
        why.append(f"vision: {p.vision}")
    if (score >= GAMEPLAY_MIN) if seen is None else seen == V_GAMEPLAY:
        ambiguous = seen is None and (score < GAMEPLAY_SURE or _text_alone(p))
        decided = None if ambiguous else "vision" if seen else _gameplay_evidence(p)
        return SourceDecision(
            p.path, p.duration, GAMEPLAY, ambiguous, ", ".join(why), score, speech, decided
        )
    if speech is not None and speech >= TALK_BAND and p.duration >= TALK_MIN_S:
        why = [f"speech {_pct(speech)} of {p.duration / 60:.0f} min", *why]
        # partial HUD evidence (not a still picture) keeps a talky game stream unsure, unless
        # the vision model saw no game
        part_hud = (
            seen is None
            and p.hud_static is not None
            and p.hud_static >= HUD_FULL / 4
            and not (p.static_share is not None and p.static_share > STILL_SHARE)
        )
        ambiguous = speech < TALK_SPEECH or part_hud
        return SourceDecision(
            p.path,
            p.duration,
            TALK,
            ambiguous,
            ", ".join(why),
            score,
            speech,
            None if ambiguous else "speech",
        )
    if speech is not None:
        why.append(f"speech {_pct(speech)}")
    reason = ", ".join(why) or "no gameplay signs"
    if seen == V_ACTION:
        return SourceDecision(p.path, p.duration, ACTION, False, reason, score, speech, "vision")
    ambiguous = seen is None and score >= GENERAL_SURE
    decided = None if ambiguous else "vision" if seen else _general_evidence(p)
    return SourceDecision(p.path, p.duration, GENERAL, ambiguous, reason, score, speech, decided)


def decide(probes: Sequence[SourceProbe], speech: Mapping[str, float] | None = None) -> Detection:
    """The project category of ``probes`` (``speech``: coverage per source path, None before
    transcription), by duration-weighted majority in two stages, so the speech coverage can
    only turn general or action into talk: first action cam (telemetry in at least
    :data:`TELEMETRY_SHARE` of the footage), gameplay or other; then, for other, talk,
    action or general. The project is ambiguous when ambiguous sources make up at least half of the
    winning category.
    """
    if not probes:
        raise ValueError("no sources to detect")
    srcs = [_source(p, None if speech is None else speech.get(p.path)) for p in probes]
    total = sum(max(s.duration, 1e-3) for s in srcs)

    def share(cats: Sequence[str], among: Sequence[SourceDecision] = srcs) -> float:
        return sum(max(s.duration, 1e-3) for s in among if s.category in cats) / total

    tel = share((ACTION_CAM,))
    if tel >= TELEMETRY_SHARE:
        cat = ACTION_CAM
    else:  # the largest group; a tie goes to other, then gameplay
        groups = [(share((TALK, GENERAL, ACTION)), 2, GENERAL), (share((GAMEPLAY,)), 1, GAMEPLAY)]
        cat = max([*groups, (tel, 0, ACTION_CAM)])[2]
        # talk against the rest, then action against general; a tie goes to general
        if cat == GENERAL:
            if share((TALK,)) > share((GENERAL, ACTION)):
                cat = TALK
            elif share((ACTION,)) > share((GENERAL,)):
                cat = ACTION
    winners = [s for s in srcs if s.category == cat]
    won = sum(max(s.duration, 1e-3) for s in winners)
    unsure = sum(max(s.duration, 1e-3) for s in winners if s.ambiguous)
    ambiguous = bool(won) and unsure >= won / 2
    scores = {c: round(share((c,)), 3) for c in CATEGORIES}
    same = (GENERAL, ACTION) if cat in (GENERAL, ACTION) else (cat,)  # one style: no disagreement
    minority = 1.0 - share(same)
    disagree = [s.path for s in srcs if s.category not in same] if minority > DISAGREE_SHARE else []
    best = max(winners, key=lambda s: s.duration)
    reason = best.reason
    if len(srcs) > 1:
        reason = f"{_pct(share((cat,)))} of the footage; {reason}"
    # a general or action source long enough for talk and not transcribed yet: talk is still open
    pending = cat in (GENERAL, ACTION) and any(
        s.category in (GENERAL, ACTION) and s.speech is None and s.duration >= TALK_MIN_S
        for s in srcs
    )
    if pending:
        reason += "; talk is checked after transcription"
    suggest = None
    if cat == GAMEPLAY:
        spoken = [s for s in winners if s.speech is not None]
        talk_s = sum(s.duration for s in spoken if (s.speech or 0) >= TALK_SPEECH)
        if spoken and talk_s >= sum(s.duration for s in winners) / 2:
            suggest = "commentary"
    weight: dict[str, float] = {}
    for s in winners:
        if s.decided_by and not ambiguous:
            weight[s.decided_by] = weight.get(s.decided_by, 0.0) + max(s.duration, 1e-3)
    return Detection(
        cat,
        ambiguous,
        reason,
        scores,
        srcs,
        pending=pending,
        disagree=disagree,
        suggest=suggest,
        decided_by=sorted(weight, key=lambda k: -weight[k]),
    )


def disagreeing(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sources of a ``detection`` record (:meth:`Detection.record`) whose category is not
    the project's, when the others make up more than :data:`DISAGREE_SHARE` of the footage (as
    :attr:`Detection.disagree`), else none."""
    cat = record.get("category")
    scores = record.get("scores")
    same = (GENERAL, ACTION) if cat in (GENERAL, ACTION) else (cat,)
    share = None
    if isinstance(scores, Mapping):
        # an old record has no ``action`` score: a missing partner counts as 0
        vals = [scores.get(c, None if c == cat else 0.0) for c in same]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            share = sum(vals)
    if (
        not isinstance(share, (int, float))
        or isinstance(share, bool)
        or 1.0 - share <= DISAGREE_SHARE
    ):
        return []
    srcs = record.get("sources")
    if not isinstance(srcs, list):
        return []
    return [s for s in srcs if isinstance(s, dict) and s.get("category") not in same]


def label(category: str) -> str:
    """Display name of a category (``Action cam``, ``Action``, ``Gameplay``, ``Talk``, ``General``)."""
    return category.replace("-", " ").capitalize()


def fallback_label(det: Detection) -> str:
    """Display name of the category whose style an ambiguous ``det`` uses
    (:func:`choose_style`): its best guess's, or General's."""
    from cutpoint.styles import auto_fallback_category  # noqa: PLC0415

    return label(auto_fallback_category(det.category))


def detect(
    sources: Sequence[Path],
    vision: bool | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    *,
    shared: FootageVision | None = None,
) -> tuple[list[SourceProbe], Detection]:
    """Probe ``sources`` (:func:`probe_source`, cached) and decide without the speech
    coverage (talk is then pending, see :attr:`Detection.pending`).

    ``vision``, ``ollama_url`` and ``ollama_model`` as for :func:`footage_vision`; ``shared``
    is a :class:`FootageVision` of the caller (the other arguments are ignored) so that two
    detections of a run ask one client; the model is freed after each detection.
    """
    logger.info("Detecting the footage category of %d source(s)", len(sources))
    if shared is not None:
        probes = [probe_source(s, shared) for s in sources]
        shared.close()
    else:
        with footage_vision(vision, ollama_url, ollama_model) as fv:
            probes = [probe_source(s, fv) for s in sources]
    return probes, decide(probes)


def choose_style(det: Detection, emphasis: str) -> tuple[str, str | None]:
    """``(style, fallback used)``: the style of ``det``'s category and ``emphasis``
    (:func:`cutpoint.styles.resolve_auto`); when the detection is ambiguous, the style of
    :func:`cutpoint.styles.auto_fallback` (also returned as the second item)."""
    from cutpoint.styles import auto_fallback, resolve_auto  # noqa: PLC0415

    if det.ambiguous:
        name = auto_fallback(emphasis, det.category)
        return name, name
    return resolve_auto(det.category, emphasis), None


def source_speech(
    sources: Sequence[Path], parts: Sequence[tuple[Path, float, Sequence[Mapping[str, Any]]]]
) -> dict[str, float]:
    """Speech coverage per source path of ``parts`` (``(source, seconds analysed, transcript
    segments)``; a source with recovered parts has several)."""
    out = {}
    for src in sources:
        mine = [(d, segs) for s, d, segs in parts if s == src]
        segments = [seg for _, segs in mine for seg in segs]
        out[str(src)] = round(speech_coverage(segments, sum(d for d, _ in mine)), 3)
    return out


def warn(det: Detection, style: str, fallback: str | None) -> None:
    """Log the decision (``style: gaming (Gameplay: static HUD 0.9%)``), with the
    ``--json`` warnings ``style_ambiguous`` and ``style_sources_disagree``."""
    from cutpoint import jsonout  # noqa: PLC0415

    logger.info("style: %s (%s: %s)", style, label(det.category), det.reason)
    if fallback is not None:
        unsure = [s.path for s in det.sources if s.ambiguous]
        message = (
            f"The footage category is not clear ({label(det.category)}?: {det.reason}); "
            f"using the {fallback_label(det)} style ({fallback}). Give --style to choose one"
        )
        logger.warning("%s", message)
        jsonout.warn("style_ambiguous", message, unsure)
    if det.disagree:
        cats = sorted({s.category for s in det.sources if s.path in det.disagree})
        message = (
            f"{len(det.disagree)} source(s) look like {', '.join(map(label, cats))}, not "
            f"{label(det.category)}; the project uses one style ({style}). Analyze them "
            "apart, or give --style"
        )
        logger.warning("%s", message)
        jsonout.warn("style_sources_disagree", message, det.disagree)
    if det.suggest:
        logger.info(
            "Speech covers most of this gameplay: --style %s keeps whole talking points",
            det.suggest,
        )
