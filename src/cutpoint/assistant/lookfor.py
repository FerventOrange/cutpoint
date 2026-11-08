"""``look_for``: rank the whole timeline by a text query with local CLIP (#529).

``find_moments`` only reads words analysis already wrote down. ``look_for("powder spray")``
compares the query with the cached frame embeddings of every source of the project (the
``events`` visual layer, about one frame per second, the same ones the visual pool and the
action-cam features read: :func:`cutpoint.events.visual.load_visual`), so it also finds footage
the scorer never proposed. Nothing is decoded and nothing is downloaded: a source whose visual
layer was never cached is skipped, and the CLIP text tower loads from local files only
(:func:`cutpoint.events.visual.load_clip`).

Per source the best frames (3-frame median, so one flicker is no peak) become windows of
:data:`MIN_LEN_S`-:data:`MAX_LEN_S` seconds, at least :data:`SPACING_S` apart. A window that an
existing moment or a row of the sequence covers (:data:`COVERED` of the shorter span) is that
moment's match. Every window carries a ``confidence``: ``strong`` from :data:`HIT_MIN`, ``weak``
from :data:`WEAK_MIN` (a weak list, never nothing for a query the footage may answer: CLIP
scores some right answers, such as "fish" in a reef, below others' wrong ones), at most
:data:`MAX_UNSURE` weak or unconfirmed rows per call. A window that nothing covers is registered
as a moment the model can view and add (:func:`new_moment`), but only a clear one
(:data:`NEW_MIN`, at most :data:`MAX_NEW` per call) is a *new moment*; the others are
unconfirmed and the rules tell the model to confirm them with view_moment or to say they are
uncertain. The model gets text only: ``match`` 0-10, ``confidence`` and the moment's row.

Without the ``events`` extra (Cutpoint lite) or without the CLIP weights the tool answers with
the install or setup command instead (:class:`LookForError`). A project none of whose sources has
cached embeddings (:func:`indexed`) is not offered the tool at all; if it is called anyway, the
page shows :class:`LookForError`'s short ``reason``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import extras, ski
from cutpoint.config import path_key
from cutpoint.events import visual as V
from cutpoint.visual_pool import _fit, _overlap

logger = logging.getLogger(__name__)

NAME = "look_for"
JSON_TOOL = f"{NAME} {{query, limit?}}"
DEFAULT_LIMIT, MAX_LIMIT = 5, 10
MAX_QUERY = 200  # characters
# CLIP ViT-L/14 image-text cosines, calibrated on Whistler day 3 (35 GoPro sources, 10k frames)
# and the scuba clips of #529: how high a query's best frame scores depends on the words as much
# as on the footage (a clear "friends on the chairlift" 0.28-0.30, a "snowboarder" 0.27, "a scuba
# diver" 0.226, but "fish" in a reef 0.204 and "a car on a road" over snow 0.220), so the bars
# are absolute and cautious. Queries about things the footage does not show peak at 0.20-0.22.
WEAK_MIN = 0.202  # the floor of the list: "fish" in a reef is 0.204, a dog over snow 0.201
HIT_MIN = 0.222  # ``strong`` from here (above the best wrong answer, 0.220)
NEW_MIN = 0.26  # a window nothing covers is a new moment from here (0.25 is bare snow)
MATCH_LO, MATCH_HI = 0.19, 0.30  # cosines that are 0 and 10 on the model's scale
MIN_LEN_S, MAX_LEN_S = 3.0, 8.0
DROP = 0.02  # a window runs while frames are within this of the peak
SPACING_S = 20.0  # between two windows of a source
PER_SOURCE = 5
MAX_NEW = 10  # new moments per call (as many as the most hits a call returns, MAX_LIMIT)
MAX_UNSURE = 4  # weak windows and unconfirmed (uncovered, not new) ones per call
COVERED = 0.5  # of the shorter span

TOOL = {
    "type": "function",
    "function": {
        "name": NAME,
        "description": (
            "Search the footage itself for what a text describes (CLIP similarity on the "
            "cached frames of every source, also footage no moment covers), best first. "
            "Use when labels and transcripts do not match. Each hit is strong or weak; a "
            "weak one is a guess to confirm (view_moment) or to call uncertain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look for, in English"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}
RULES = f"""

{NAME}(query) compares the query with the footage itself (CLIP on the cached frames of every \
source) and lists the best windows with match 0-10 and confidence strong or weak. Use it when \
find_moments, labels, seen, cues and transcripts do not match what the user wants to see (powder \
spray, a sunset, a red jacket), with a short English description of the picture. A clear window \
nobody covered comes back as a new moment (new: true) that you can add; one that is weak or \
unconfirmed is a guess. Similarity is not certainty: say a match is likely or possible, never \
sure, and name the moment so the user can check it. If {NAME} says it is not available, tell the \
user what it says."""
WEAK_IMAGES = f"""
A weak or unconfirmed {NAME} window is not confirmed: check it with view_moment before you call \
it a match. propose_edit refuses to add an unconfirmed window until view_moment has shown it; \
the user sees each one as "possible" in the chat and can add it themself."""
WEAK_TEXT = f"""
A weak or unconfirmed {NAME} window is not confirmed and you cannot look at it: tell the user \
it is uncertain and name it. propose_edit refuses to add an unconfirmed window; the user sees \
each one as "possible" in the chat and can add it themself."""


def rules(images: bool) -> str:
    """:data:`RULES` with what to do about weak windows: confirm them with ``view_moment`` when
    the model takes images (``images``), else say they are uncertain."""
    return RULES + (WEAK_IMAGES if images else WEAK_TEXT)


NO_INDEX = (
    "Visual search isn't available for this project: its analysis has no frame index "
    "(analyze --visual-index writes it)"
)


NO_LITE = "Visual search isn't in Cutpoint lite"
NO_WEIGHTS = "Visual search needs the CLIP weights (cutpoint setup --only visual)"


class LookForError(Exception):
    """The search cannot run; the message says what to do and goes to the model. ``reason`` is a
    short plain sentence for the user's page (None: the page says only that it failed)."""

    def __init__(self, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


def match_of(cos: float) -> int:
    """A cosine on the model's 0-10 scale (:data:`MATCH_LO`-:data:`MATCH_HI`)."""
    return round(10 * min(1.0, max(0.0, (cos - MATCH_LO) / (MATCH_HI - MATCH_LO))))


@dataclass
class Hit:
    """A window of a source that matches the query."""

    source: str
    start: float
    end: float
    cos: float  # of the peak frame

    @property
    def match(self) -> int:
        """0-10 (:data:`MATCH_LO`-:data:`MATCH_HI`)."""
        return match_of(self.cos)

    @property
    def confidence(self) -> str:
        """``strong`` from :data:`HIT_MIN`, else ``weak``."""
        return "strong" if self.cos >= HIT_MIN else "weak"

    @property
    def clear(self) -> bool:
        """Clear enough to be a new moment (:data:`NEW_MIN`)."""
        return self.cos >= NEW_MIN


@dataclass
class Found:
    """What a search gives: ``hits`` best first (each with the id it covers, or None), the
    number of sources searched and of all, and the best ``match`` of any frame."""

    hits: list[tuple[Hit, str | None]]
    searched: int
    total: int
    best: int


def indexed(sources: Sequence[str]) -> bool:
    """Whether any of ``sources`` has cached frame embeddings (the file only is looked for:
    nothing is read), so :data:`NAME` has something to search."""
    for source in sources:
        try:
            if V.cache_path(Path(source), V.VisualParams()).exists():
                return True
        except OSError:  # offline footage: its cache key needs the file's size and mtime
            continue
    return False


def unavailable() -> str | None:
    """Why the search cannot run here: the lite edition's message with the install command
    (:func:`cutpoint.extras.missing_module_message`), or None."""
    for module in ("torch", "open_clip"):
        if importlib.util.find_spec(module) is None:
            return extras.missing_module_message(module)
    return None


def moment_id(source: str, start: float, end: float) -> str:
    """The id ``cutpoint.ui.moments_view`` would give an analysis moment of this span."""
    key = f"{os.path.realpath(source)}|{float(start)}|{float(end)}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def new_moment(hit: Hit, query: str) -> dict[str, Any]:
    """The candidate (``moments_view`` shape) for a hit nothing covers; ``pool`` marks where it
    came from and ``score`` is ``match`` x 10, for display."""
    start, end = round(hit.start, 2), round(hit.end, 2)
    return {
        "id": moment_id(hit.source, start, end),
        "source": hit.source,
        "start": start,
        "end": end,
        "label": " ".join(query.split())[:60],
        "score": hit.match * 10,
        "pool": NAME,
    }


def _windows(source: str, t: np.ndarray, sims: np.ndarray, duration: float | None) -> list[Hit]:
    """The windows of one source: peaks of ``sims`` (at frame times ``t``) from the best on,
    :data:`SPACING_S` apart, each cut to :data:`MIN_LEN_S`-:data:`MAX_LEN_S`."""
    sims = ski.smooth3(sims)
    end_of = duration or float(t[-1]) + 1.0
    out: list[Hit] = []
    peaks: list[float] = []
    for i in np.argsort(-sims):
        if sims[i] < WEAK_MIN or len(out) >= PER_SOURCE:
            break
        at = float(t[i])
        if any(abs(at - u) < SPACING_S for u in peaks):
            continue
        peaks.append(at)
        lo = hi = int(i)
        while lo > 0 and sims[lo - 1] >= sims[i] - DROP:
            lo -= 1
        while hi < len(t) - 1 and sims[hi + 1] >= sims[i] - DROP:
            hi += 1
        a, b = _fit(float(t[lo]), float(t[hi]) + 1.0, at, MIN_LEN_S, MAX_LEN_S, end_of)
        out.append(Hit(source, a, b, float(sims[i])))
    return out


class Finder:
    """The search over a project's sources. The cached embeddings are read once per finder
    (one chat turn), so a second query costs a matrix product."""

    def __init__(self, sources: Sequence[str]) -> None:
        self.sources = list(sources)
        self._frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    def _load(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._frames is None:
            frames = {}
            for source in self.sources:
                try:
                    data = V.load_visual(Path(source))
                except (OSError, ValueError, KeyError, zipfile.BadZipFile) as err:
                    logger.warning("look_for: %s: visual layer unreadable (%s)", source, err)
                    continue
                if data is not None and len(data.t):
                    frames[source] = (data.t.astype(float), data.mean_emb)
            self._frames = frames
        return self._frames

    def search(
        self,
        query: str,
        limit: int,
        covering: Sequence[Mapping[str, Any]],
        durations: Mapping[str, float | None],
    ) -> Found:
        """The best windows of every source with cached embeddings.

        Args:
            query: What to look for.
            limit: At most this many hits.
            covering: Moments and sequence rows (``id``, ``source``, ``start``, ``end``) that
                a window can match.
            durations: Known source durations (windows stay inside them).

        Raises:
            LookForError: In the lite edition, without cached embeddings or without the CLIP
                weights.
        """
        if (why := unavailable()) is not None:
            raise LookForError(why, NO_LITE)
        frames = self._load()
        if not frames:
            raise LookForError(
                "no source of this project has cached frame embeddings (analyze writes them "
                "with its visual step for action, sport and action-cam footage, and for other "
                "styles with --visual-index), so there is nothing to look through; use "
                "find_moments instead",
                NO_INDEX,
            )
        try:
            text = V.text_embeddings((query,), save=False)[0]
        except V.ClipWeightsMissing:
            raise LookForError(
                "the CLIP weights are not on this machine; the user can get them with: "
                "cutpoint setup --only visual",
                NO_WEIGHTS,
            ) from None
        hits: list[Hit] = []
        best = 0.0
        for source, (t, emb) in frames.items():
            sims = emb @ text.astype(np.float32)
            best = max(best, float(sims.max()))
            hits += _windows(source, t, sims, durations.get(source))
        hits.sort(key=lambda h: -h.cos)
        out: list[tuple[Hit, str | None]] = []
        seen: set[str] = set()
        new = unsure = 0
        for hit in hits:
            mine = [c for c in covering if path_key(str(c["source"])) == path_key(hit.source)]
            cover = _covering(hit, mine)
            if cover is not None and cover in seen:
                continue
            is_new = cover is None and hit.clear
            if is_new and new >= MAX_NEW:
                continue
            sure = hit.confidence == "strong" and (cover is not None or is_new)
            if not sure and unsure >= MAX_UNSURE:
                continue
            if cover is not None:
                seen.add(cover)
            new += is_new
            unsure += not sure
            out.append((hit, cover))
            if len(out) >= limit:
                break
        return Found(out, len(frames), len(self.sources), match_of(best))


def _covering(hit: Hit, mine: Sequence[Mapping[str, Any]]) -> str | None:
    """The id of the item of ``mine`` that covers ``hit`` most, else None."""
    best, found = 0.0, None
    for c in mine:
        a, b = float(c["start"]), float(c["end"])
        over = _overlap(hit.start, hit.end, a, b)
        if over >= COVERED * min(hit.end - hit.start, b - a) and over > best:
            best, found = over, str(c["id"])
    return found
