"""HUD fit check (#822): does a source's HUD text land where a preset's game shows it?

The game key of a source is first guessed from its folder or recorder file name
(:func:`cutpoint.events.regions.first_guess`), so games recorded into one folder share
a guess. Once a preset exists for the guess (found, or discovered in this run), each
source whose key was guessed is checked against it; one whose text lands elsewhere gets
a key and a discovery of its own (:func:`cutpoint.events.pipeline._discover`).

Only text along the screen edges counts (a HUD hugs them, the game view fills the
centre): OCR boxes of at least the discovery's minimum score with two letters or digits,
centred in the outer cells of discovery's coarse 8 x 6 menu grid and outside the big-text
region (:func:`edge_boxes`). The signal is a grid of :data:`GRID_W` x :data:`GRID_H` cells
over the reference frame: per cell, the share of gameplay frames with an edge box centred
in it (:meth:`Profile.grid`). The preset's regions play no part.

A source fits a reference when the cosine of the two grids is at least :data:`TAU_GRID`.
The reference is the other sources of the same discovery run (:func:`split_group`) or the
preset's ``discovery.fit_ref``, which discovery writes (:meth:`Profile.ref`). A preset
without a grid ``fit_ref`` (none, or the per-region one of an earlier version) decides
nothing: its sources keep the guess until a discovery writes one (:func:`against_ref`).
Too little evidence (fewer than :data:`MIN_FRAMES` gameplay frames or :data:`MIN_BOXES`
edge boxes) decides nothing either: the guess stays.

No game names and no game-specific parsing: only positions of text boxes.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint.events.discover import (
    MENU_CELL_H,
    MENU_CELL_W,
    REF_H,
    REF_W,
    DiscoverParams,
    _ocr_frames,
    plan_bursts,
)
from cutpoint.events.regions import Region, bigtext_region

# the edge-cell grid of the signal (columns, rows over REF_W x REF_H)
GRID_W, GRID_H = 16, 12
# A source fits when cos(grid, reference grid) >= TAU_GRID. Provisional: on 9 sources of 5
# games, same-game pairs scored 0.62-0.99 and cross-game pairs at most 0.51.
TAU_GRID = 0.57
# evidence: fewer gameplay frames or edge boxes than these decide nothing
MIN_FRAMES = 4
MIN_BOXES = 15
# a source with fewer gameplay frames than TOP_UP_FRAMES in the run's discovery gets
# SAMPLE_BURSTS bursts of its own (DiscoverParams.burst_frames frames each)
TOP_UP_FRAMES = 8
SAMPLE_BURSTS = 2


def _inside(b: dict[str, Any], r: Region) -> bool:
    cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
    return r.x <= cx <= r.x + r.w and r.y <= cy <= r.y + r.h


def edge_boxes(frame: dict[str, Any], min_score: float) -> list[dict[str, Any]]:
    """The boxes of an OCR record that count: score at least ``min_score``, two letters
    or digits, centred in an outer cell of the 8 x 6 menu grid and not in the big-text
    region."""
    big = bigtext_region()
    out = []
    for b in frame.get("boxes", []):
        if float(b.get("score", 0)) < min_score:
            continue
        if sum(c.isalnum() for c in str(b.get("text", ""))) < 2:
            continue
        gx = int((b["x0"] + b["x1"]) / 2 // MENU_CELL_W)
        gy = int((b["y0"] + b["y1"]) / 2 // MENU_CELL_H)
        if 1 <= gx <= 6 and 1 <= gy <= 4:
            continue
        if not _inside(b, big):
            out.append(b)
    return out


def _cell(b: dict[str, Any]) -> str | None:
    """The grid cell (``"x,y"``) of a box's centre; None outside the reference frame."""
    cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
    if not (0 <= cx < REF_W and 0 <= cy < REF_H):
        return None
    return f"{int(cx * GRID_W / REF_W)},{int(cy * GRID_H / REF_H)}"


@dataclass(frozen=True)
class Profile:
    """Counts of one or more sources' gameplay frames over the edge-cell grid.

    Attributes:
        frames: Gameplay frames.
        boxes: Edge boxes (:func:`edge_boxes`).
        seen: Per grid cell (``"x,y"``): frames with an edge box centred in it (cells
            with one only).
    """

    frames: int = 0
    boxes: int = 0
    seen: tuple[tuple[str, int], ...] = ()

    def __add__(self, other: Profile) -> Profile:
        seen = dict(self.seen)
        for cell, n in other.seen:
            seen[cell] = seen.get(cell, 0) + n
        return Profile(self.frames + other.frames, self.boxes + other.boxes, tuple(seen.items()))

    def grid(self) -> dict[str, float]:
        """Per cell, the share of gameplay frames with an edge box in it."""
        return {cell: n / self.frames if self.frames else 0.0 for cell, n in self.seen}

    def enough(self) -> bool:
        """Whether the counts can decide anything."""
        return bool(self.seen) and self.frames >= MIN_FRAMES and self.boxes >= MIN_BOXES

    def ref(self) -> dict[str, Any]:
        """The ``discovery.fit_ref`` of a preset made from these frames."""
        return {
            "cells": [GRID_W, GRID_H],
            "grid": {cell: round(v, 3) for cell, v in sorted(self.grid().items())},
            "frames": self.frames,
            "boxes": self.boxes,
        }


def profile(frames: Iterable[dict[str, Any]], min_score: float) -> Profile:
    """The :class:`Profile` of gameplay ``frames`` (OCR records); each cell counts once
    per frame."""
    n = boxes = 0
    seen: dict[str, int] = {}
    for f in frames:
        n += 1
        hit: set[str] = set()
        for b in edge_boxes(f, min_score):
            boxes += 1
            cell = _cell(b)
            if cell is not None:
                hit.add(cell)
        for cell in hit:
            seen[cell] = seen.get(cell, 0) + 1
    return Profile(n, boxes, tuple(seen.items()))


def cosine(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Cosine of two grids over the union of their cells; None when either is all zero."""
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return None
    return sum(a.get(k, 0.0) * v for k, v in b.items()) / (na * nb)


@dataclass(frozen=True)
class Fit:
    """A conclusive fit test: ``ok`` and the numbers it was decided on. ``score`` is the
    grid cosine; ``ref`` is ``run`` (other sources of this run) or ``preset``."""

    ok: bool
    score: float
    frames: int
    ref: str

    def record(self) -> dict[str, Any]:
        """The ``fit`` of a ``game_keys.json`` entry."""
        return {"score": round(self.score, 3), "frames": self.frames, "ref": self.ref}


def _relative(p: Profile, grid: dict[str, float], ref: str) -> Fit | None:
    cos = cosine(p.grid(), grid)
    if cos is None:
        return None
    return Fit(cos >= TAU_GRID, cos, p.frames, ref)


def valid_ref(ref: Any) -> bool:
    """Whether ``ref`` is a usable ``discovery.fit_ref``: a grid of this signal (not the
    per-region ``occ`` of an earlier version, nor one of other cells), each value a share
    from 0 to 1 (NaN is not)."""
    if not isinstance(ref, dict) or ref.get("cells") != [GRID_W, GRID_H]:
        return False
    grid = ref.get("grid")
    return isinstance(grid, dict) and all(
        _valid_cell(k) and isinstance(v, int | float) and not isinstance(v, bool) and 0 <= v <= 1
        for k, v in grid.items()
    )


def _valid_cell(key: Any) -> bool:
    """Whether ``key`` names a cell of the grid (``"x,y"``)."""
    m = re.fullmatch(r"([0-9]+),([0-9]+)", key) if isinstance(key, str) else None
    return m is not None and int(m[1]) < GRID_W and int(m[2]) < GRID_H


def against_ref(p: Profile, ref: Any) -> Fit | None:
    """Fit of a source's profile to a preset's ``fit_ref``. None: too little evidence, or
    no grid ``fit_ref`` (missing, or of an earlier version) to compare with."""
    if not p.enough() or not valid_ref(ref):
        return None
    return _relative(p, {k: float(v) for k, v in ref["grid"].items()}, "preset")


def against_run(p: Profile, other: Profile) -> Fit | None:
    """Fit of a source's profile to another profile of this run (``ref`` ``run``). None:
    too little evidence in ``p``."""
    if not p.enough():
        return None
    return _relative(p, other.grid(), "run")


def split_group(
    profiles: dict[str, Profile], durations: dict[str, float], fixed: frozenset[str] = frozenset()
) -> dict[str, Fit | None]:
    """Which sources of one discovery run fit together.

    The leader is the source with the most gameplay frames (enough evidence needed); the
    group profile starts as the leader plus the ``fixed`` sources (decided before, never
    tested). The others, longest first, join when their grid fits the group's, which then
    pools them too.

    Returns:
        Per source not in ``fixed``: its :class:`Fit` (``ref`` ``run``), or None when
        there is too little evidence (its own, or no source has enough).
    """
    order = sorted(profiles, key=lambda k: (-durations.get(k, 0.0), k))
    usable = [k for k in order if profiles[k].enough()]
    out: dict[str, Fit | None] = {k: None for k in order if k not in fixed}
    if not usable:
        return out
    leader = max(usable, key=lambda k: profiles[k].frames)  # ties: the longest
    group = profiles[leader]
    for k in fixed:
        if k != leader and k in profiles:
            group += profiles[k]
    if leader not in fixed:
        out[leader] = against_run(profiles[leader], group)
    for k in order:
        if k == leader or k in fixed or not profiles[k].enough():
            continue
        fit = against_run(profiles[k], group)
        out[k] = fit
        if fit is not None and fit.ok:
            group += profiles[k]
    return out


def sample_frames(
    sources: list[Any], out_dir: Path, params: DiscoverParams
) -> list[dict[str, Any]]:
    """OCR :data:`SAMPLE_BURSTS` bursts of each source into ``out_dir`` (``frames/`` and
    ``ocr.jsonl``, cached like discovery's); the OCR model loads only when a frame needs
    reading. Folders left empty (nothing decoded) are removed, ``out_dir``'s parent too."""
    bursts = [b for s in sources for b in plan_bursts([s], SAMPLE_BURSTS)]
    if not bursts:
        return []
    got = _ocr_frames(bursts, params, out_dir / "frames", out_dir / "ocr.jsonl")
    for folder in (out_dir / "frames", out_dir, out_dir.parent):  # nothing read: no empty folders
        with contextlib.suppress(OSError):
            folder.rmdir()  # only when empty
    return got


__all__ = [
    "Fit",
    "Profile",
    "against_ref",
    "against_run",
    "cosine",
    "edge_boxes",
    "profile",
    "sample_frames",
    "split_group",
]
