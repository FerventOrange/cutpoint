"""Crop handling for bordered recordings (e.g. a stream layout with black bars).

Specs accepted by ``cutpoint clips --crop``, per-segment ``"crop"`` and the
``project.json`` ``"crop"`` map:

- ``none``: no crop.
- ``auto``: ffmpeg ``cropdetect`` on a few frames sampled across the source,
  cached per source in its cache folder (``crop.json``).
- ``W:H:X:Y``: explicit crop box in source pixels (ffmpeg ``crop`` order).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint import cache
from cutpoint.ffmpeg_utils import MediaInfo, cropdetect_at
from cutpoint.utils import matches_glob

logger = logging.getLogger(__name__)

CROP_CACHE = "crop.json"
CROP_CACHE_VERSION = 1
SAMPLE_FRACTIONS = (0.12, 0.3, 0.45, 0.62, 0.8, 0.93)
NEGLIGIBLE = 0.98  # a detected box covering >= 98% of both dimensions = no border
ASPECT_TRIM_MAX = 0.06  # trim the box to the output aspect if within 6%, else pad


@dataclass(frozen=True)
class Crop:
    """Crop box in source pixels."""

    w: int
    h: int
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.w}:{self.h}:{self.x}:{self.y}"

    def fit_aspect(self, aspect: float) -> Crop:
        """Trim the box (centred) to ``aspect`` if it is within ``ASPECT_TRIM_MAX``.

        That way a slightly-off box (1364x750 is 1.819 vs 16:9's 1.778) fills
        the frame exactly instead of getting thin bars. Larger mismatches are
        left as they are and get padded.
        """
        box = self.w / self.h
        if abs(box - aspect) / aspect > ASPECT_TRIM_MAX:
            return self
        if box > aspect:
            w = int(round(self.h * aspect / 2) * 2)
            return Crop(w, self.h, self.x + (self.w - w) // 2, self.y)
        h = int(round(self.w / aspect / 2) * 2)
        return Crop(self.w, h, self.x, self.y + (self.h - h) // 2)


def parse_spec(spec: str | None) -> str | Crop | None:
    """Parse a crop spec into ``None`` (no crop), ``"auto"`` or a :class:`Crop`.

    Raises:
        ValueError: If the spec is malformed.
    """
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if text in ("", "none", "off", "no"):
        return None
    if text == "auto":
        return "auto"
    try:
        w, h, x, y = (int(v) for v in text.replace("x", ":").replace("+", ":").split(":"))
    except ValueError:
        raise ValueError(f"Invalid crop {spec!r}; use auto, none or W:H:X:Y") from None
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError(f"Invalid crop {spec!r}: sizes must be positive")
    return Crop(w, h, x, y)


def detect(src: Path, info: MediaInfo, readable_until: float | None = None) -> Crop | None:
    """Detect black borders with ``cropdetect`` on sampled frames (cached).

    Samples are combined by majority (the most common box if at least half
    agree), else by union, so a dark scene that looks "more bordered" cannot
    shrink the box. A box that is essentially the full frame means no crop.

    Args:
        src: Source video.
        info: Its probe result.
        readable_until: Sample only before this time (damaged files).

    Returns:
        The crop box, or None if the source has no borders.
    """
    cfile = cache.source_dir(src) / CROP_CACHE
    cached = cache.load_json(cfile)
    if cached and cached.get("version") == CROP_CACHE_VERSION:
        return Crop(*cached["crop"]) if cached.get("crop") else None
    span = min(info.duration, readable_until or info.duration)
    boxes = []
    for frac in SAMPLE_FRACTIONS:  # one attempt each: never retry a bad spot
        box = cropdetect_at(src, frac * span)
        if box:
            boxes.append(box)
    crop: Crop | None = None
    if boxes:
        ((mode, count),) = Counter(boxes).most_common(1)
        if count * 2 >= len(boxes):
            w, h, x, y = mode
        else:
            x = min(b[2] for b in boxes)
            y = min(b[3] for b in boxes)
            w = max(b[2] + b[0] for b in boxes) - x
            h = max(b[3] + b[1] for b in boxes) - y
        if w < NEGLIGIBLE * info.width or h < NEGLIGIBLE * info.height:
            crop = Crop(w, h, x, y)
    logger.info(
        "cropdetect %s: %d/%d samples -> %s",
        src.name,
        len(boxes),
        len(SAMPLE_FRACTIONS),
        crop or "no border",
    )
    cache.save_json(
        cfile,
        {
            "version": CROP_CACHE_VERSION,
            "crop": [crop.w, crop.h, crop.x, crop.y] if crop else None,
            "samples": [list(b) for b in boxes],
        },
    )
    return crop


def project_crop(project_cfg: dict[str, Any] | None, src: Path) -> str | None:
    """Per-source crop spec from ``project.json`` (``{"crop": {pattern: spec}}``).

    Patterns match the file name or the full path, case-insensitively, with
    shell-style wildcards (``"Stream*.mp4"``).
    """
    table = (project_cfg or {}).get("crop") or {}
    for pattern, spec in table.items():
        if matches_glob(src, (pattern,)):
            return spec
    return None
