"""HUD fit check (#822): does a source's HUD text land in a preset's regions?

The game key of a source is first guessed from its folder or recorder file name
(:func:`cutpoint.events.regions.first_guess`), so games recorded into one folder share
a guess. Once a preset exists for the guess (found, or discovered in this run), each
source whose key was guessed is checked against it; one whose text lands elsewhere gets
a key and a discovery of its own (:func:`cutpoint.events.pipeline._discover`).

Only text along the screen edges counts (a HUD hugs them, the game view fills the
centre): OCR boxes of at least the discovery's minimum score with two letters or digits,
centred in the outer cells of discovery's coarse 8 x 6 menu grid and outside the big-text
region (:func:`edge_boxes`). Per source and region set:

* ``inside``: the share of those boxes inside a region;
* ``occ``: per region, the share of gameplay frames with such a box in it.

A source fits a reference when the cosine of its ``occ`` and the reference's is at
least :data:`TAU_COS` and its ``inside`` at least :data:`INSIDE_SHARE` of the
reference's. The reference is the other sources of the same discovery run
(:func:`split_group`) or the preset's ``discovery.fit_ref``, which discovery writes; a
preset without one needs ``inside`` of :data:`TAU_ABS` and a box in a feed or notice
region, as does one whose ``fit_ref`` names regions it no longer has (:func:`against_ref`).
Too little evidence (fewer than :data:`MIN_FRAMES`
gameplay frames or :data:`MIN_BOXES` edge boxes) decides nothing: the guess stays.

No game names and no game-specific parsing: only positions of text boxes.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint.events.discover import (
    MENU_CELL_H,
    MENU_CELL_W,
    DiscoverParams,
    _ocr_frames,
    plan_bursts,
)
from cutpoint.events.regions import Region, bigtext_region

# A source fits when cos(occ, reference occ) >= TAU_COS and its inside share is at least
# INSIDE_SHARE of the reference's; against a preset without fit_ref: inside >= TAU_ABS and
# a box in a feed or notice region. Starting values, to be calibrated on footage.
TAU_COS = 0.5
TAU_ABS = 0.35
INSIDE_SHARE = 0.5
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


@dataclass(frozen=True)
class Profile:
    """Counts of one or more sources' gameplay frames over a region set.

    Attributes:
        frames: Gameplay frames.
        boxes: Edge boxes (:func:`edge_boxes`).
        inside: Edge boxes inside a region.
        feed_hits: Edge boxes inside a ``feed`` or ``notice`` region.
        seen: Per region name: frames with an edge box in it (every region, also 0).
    """

    frames: int = 0
    boxes: int = 0
    inside: int = 0
    feed_hits: int = 0
    seen: tuple[tuple[str, int], ...] = ()

    def __add__(self, other: Profile) -> Profile:
        seen = dict(self.seen)
        for name, n in other.seen:
            seen[name] = seen.get(name, 0) + n
        return Profile(
            self.frames + other.frames,
            self.boxes + other.boxes,
            self.inside + other.inside,
            self.feed_hits + other.feed_hits,
            tuple(seen.items()),
        )

    @property
    def inside_share(self) -> float:
        return self.inside / self.boxes if self.boxes else 0.0

    def occ(self) -> dict[str, float]:
        return {name: n / self.frames if self.frames else 0.0 for name, n in self.seen}

    def enough(self) -> bool:
        """Whether the counts can decide anything."""
        return bool(self.seen) and self.frames >= MIN_FRAMES and self.boxes >= MIN_BOXES

    def ref(self) -> dict[str, Any]:
        """The ``discovery.fit_ref`` of a preset made from these frames."""
        return {
            "inside": round(self.inside_share, 3),
            "occ": {name: round(v, 3) for name, v in self.occ().items()},
        }


def profile(
    frames: Iterable[dict[str, Any]], regions: Iterable[Region], min_score: float
) -> Profile:
    """The :class:`Profile` of gameplay ``frames`` (OCR records) over ``regions`` (a
    ``bigtext`` region is left out: its boxes never count)."""
    regs = [r for r in regions if r.kind != "bigtext"]
    n = boxes = inside = feed = 0
    seen = dict.fromkeys((r.name for r in regs), 0)
    for f in frames:
        n += 1
        hit: set[str] = set()
        for b in edge_boxes(f, min_score):
            boxes += 1
            got = [r for r in regs if _inside(b, r)]
            inside += bool(got)
            feed += any(r.kind in ("feed", "notice") for r in got)
            hit.update(r.name for r in got)
        for name in hit:
            seen[name] += 1
    return Profile(n, boxes, inside, feed, tuple(seen.items()))


def cosine(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Cosine of two occupancy vectors over ``b``'s regions; None when ``b`` is all zero
    (no reference to compare with), 0 when ``a`` is."""
    nb = math.sqrt(sum(v * v for v in b.values()))
    if nb == 0:
        return None
    na = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in b))
    if na == 0:
        return 0.0
    return sum(a.get(k, 0.0) * v for k, v in b.items()) / (na * nb)


@dataclass(frozen=True)
class Fit:
    """A conclusive fit test: ``ok`` and the numbers it was decided on. ``score`` is the
    cosine (``ref`` ``run`` or ``preset``) or the inside share (``absolute``)."""

    ok: bool
    score: float
    inside: float
    frames: int
    ref: str

    def record(self) -> dict[str, Any]:
        """The ``fit`` of a ``game_keys.json`` entry."""
        return {
            "score": round(self.score, 3),
            "inside": round(self.inside, 3),
            "frames": self.frames,
            "ref": self.ref,
        }


def _relative(p: Profile, occ: dict[str, float], inside: float, ref: str) -> Fit | None:
    cos = cosine(p.occ(), occ)
    if cos is None:
        return None
    ok = cos >= TAU_COS and p.inside_share >= INSIDE_SHARE * inside
    return Fit(ok, cos, p.inside_share, p.frames, ref)


def valid_ref(ref: Any) -> bool:
    """Whether ``ref`` is a usable ``discovery.fit_ref``."""
    return (
        isinstance(ref, dict)
        and isinstance(ref.get("inside"), int | float)
        and isinstance(ref.get("occ"), dict)
        and all(isinstance(v, int | float) for v in ref["occ"].values())
    )


def against_ref(p: Profile, ref: dict[str, Any] | None) -> Fit | None:
    """Fit of a source's profile (over a preset's regions) to the preset's ``fit_ref``,
    or, without a usable one (also one naming a region the preset no longer has), to the
    absolute rule. None: too little evidence."""
    if not p.enough():
        return None
    names = {name for name, _ in p.seen}
    # a fit_ref of regions the preset no longer has (renamed or replaced by hand) says nothing
    if valid_ref(ref) and set(map(str, ref["occ"])) <= names:
        assert ref is not None
        occ = {str(k): float(v) for k, v in ref["occ"].items()}
        return _relative(p, occ, float(ref["inside"]), "preset")
    ok = p.inside_share >= TAU_ABS and p.feed_hits >= 1
    return Fit(ok, p.inside_share, p.inside_share, p.frames, "absolute")


def split_group(
    profiles: dict[str, Profile], durations: dict[str, float], fixed: frozenset[str] = frozenset()
) -> dict[str, Fit | None]:
    """Which sources of one discovery run fit together.

    The leader is the source with the most gameplay frames (enough evidence needed); the
    group profile starts as the leader plus the ``fixed`` sources (decided before, never
    tested). The others, longest first, join when they fit the group profile, which then
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
        out[leader] = _relative(profiles[leader], group.occ(), group.inside_share, "run")
    for k in order:
        if k == leader or k in fixed or not profiles[k].enough():
            continue
        fit = _relative(profiles[k], group.occ(), group.inside_share, "run")
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
    "cosine",
    "edge_boxes",
    "profile",
    "sample_frames",
    "split_group",
]
