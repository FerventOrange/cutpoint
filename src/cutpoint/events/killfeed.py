"""Layer 1b: on-screen events (kill feed, round banners, notices) via OCR.

One sequential ffmpeg pass per source decodes only the bounding box of the
game's HUD regions at a low frame rate (the probed GPU decoder when it works) and
pipes raw BGR frames to numpy. Each region is OCR'd with RapidOCR (CPU) only
when its gate lets it through: the text layout RapidOCR detects changed ("det", which
skips the vast majority of frames) or on every frame the region is due ("read").

Events: ``{t, end, type, text, confidence, region, ...}`` with type one of
``kill`` (and the other feed verbs), ``banner``, ``notice``, ``bigtext``, ``multikill`` and
``score`` (a "score" region's number stepped by one,
:class:`~cutpoint.events.score_readout.ScoreTracker`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.errors import FFmpegError, is_out_of_memory
from cutpoint.events import ocr_raw
from cutpoint.events.common import Source, layer_cache_path
from cutpoint.events.menu_rules import _union
from cutpoint.events.ocr_engine import OcrEngine
from cutpoint.events.ocr_read import (
    DET_SKIP_MAX_DIFF,
    DET_SKIP_SCALE,
    DET_SKIP_TILE,
    MIN_TEXT_SCORE,
    FrameRec,
    OcrInterpretation,
    OcrReader,
)
from cutpoint.events.regions import (
    REF_H,
    HudPreset,
    Region,
    game_key_for_path,
    load_preset,
    scaled,
)
from cutpoint.events.tracker import FUZZY, PERSIST_S
from cutpoint.ffmpeg_utils import READ_ERROR_RE, decode_raw, decoder_for, probe
from cutpoint.imageio import write_png

logger = logging.getLogger(__name__)


@dataclass
class OcrParams:
    """Tunable parameters for :func:`detect_ocr_events` (all go into the cache key)."""

    fps: float = 3.0
    game: str = "auto"
    player_tags: tuple[str, ...] = ()  # config.json ``player_tags`` (or ``--player``)
    hwaccel: bool = True
    project: Path | None = None  # for project-local presets (not part of the key)
    # skip the det gate of a region that has not changed (DET_SKIP_*); config.json
    # ``ocr_det_skip`` or ``--ocr-det-skip``. In the raw read's key, so in the key through it
    det_skip: bool = False

    def key(self, game: str | None, preset: HudPreset | None = None) -> dict[str, Any]:
        """Cache key (hwaccel does not change results, so it is left out).

        :func:`detect_ocr_events` adds the hash of the raw read's key (``"raw"``,
        which includes the RapidOCR version), so results are read again after a
        RapidOCR upgrade.
        """
        key = {
            "fps": self.fps,
            "game": game,
            "tags": list(self.player_tags),
            "min_text_score": MIN_TEXT_SCORE,
            "fuzzy": FUZZY,
            "persist_s": PERSIST_S,
            "regions": [r.key() for r in (preset.all_regions() if preset else ())],
            "v": 4,  # 4: no built-in presets, no ``extra``
        }
        if preset is not None:
            key["preset"] = preset.key() | {"interp": GENERIC_VERSION}
        return key


# bump when the generic (JSON preset) interpretation changes; 7: menu frames give no
# events (:func:`menu_rules`), multi-word phrases and feed kill words match whole words only;
# 8: static screens are menus too (:func:`static_frames`); 9: menu frames keep feed sightings
# and kills, a static screen without new menu text must hold still MENU.static_bare_min_s;
# 10: new menu text is MENU.static_text_lines texts that stay for MENU.static_text_share of the part;
# 11: screens with MENU.text_lines lines of new text that stay, MENU.text_together first read together,
# are menus too (:func:`text_frames`)
# 12: "row_notices" feeds without team colours read icon kill lines ("name [icon] name",
# FeedParser._icon_line) and other rows as notices; icon kill events keep their most complete
# names and are dropped on menu frames
# 13: menu frames are recorded as stats["menu_spans"] (#440)
# 14: gaps up to MENU.menu_fill_s between box- or text-rule menu frames are menus too (#409)
# 15: a static screen without new menu text needs menu frames within MENU.static_anchor_s (#554)
# 16: feeds learned row forms from the footage, a clock-stamped log's wrapped rows are joined,
# the default kill words are generic only and player tags tolerate OCR variants
# (:func:`is_player`)
# 17: feeds learn nothing from the footage, so no kills learned at 16 are reused
# 18: a discovered feed region that reads like stream chat (:func:`chat_label`) gives no events
# 19: a chat row (:func:`_is_chat`) gives no kill, assist or crash on any feed, not only in a log
# 20: reads of one kill row are one kill (:func:`merge_kill_reads`), a banner text that stays on a
# screen is one banner (:func:`collapse_banner_bursts`), a recurring system line is no kill
# (:meth:`Interpreter.drop_notice_kills`) and tags match tighter and in glued names
# (:func:`player_side`)
# 21: banner bursts remove a phrase OCR reads the same wrong way at its best fuzzy
# match (:func:`collapse_banner_bursts`)
# 22: big-text lines that come back as short pop-ups all through a source are flagged
# ``announce`` (:func:`learn_announcements`), short lines with digits included
# 23: text-rule menu frames are further than MENU.static_feed_guard_s from feed activity (#858)
# 24: the rows a feed reads as notices are one line of new menu text at most, however many
# are on screen (:func:`feed_stacks`, #891)
# 25: ... but the rows a feed reads first in one frame are lines of their own (a screen's text in
# the feed's area, #891)
# 26: text in any script is compared by its letters and digits, not only A-Z and 0-9 (:func:`norm`,
# #1085 A137); ASCII text gives the same events, replayed from the raw OCR cache
# 27: the default kill words are killed, eliminated, destroyed and assist only, and a feed gets
# the vehicle groups of its names only with the preset's own ``kill_words`` (#1085 A143)
GENERIC_VERSION = 27
# Reads of a source whose part gave no frames before the cached result is kept as it is.
MAX_INCOMPLETE_READS = 2


# --------------------------------------------------------------------------- decoding


class OcrCancelled(BaseException):
    """A read with a ``cancel`` event was stopped: nothing after the last finished raw
    chunk is kept. A BaseException, so the read's ``except RuntimeError`` (a failed
    decode pass) never takes it for a decode error."""


def ocr_vf(fps: float, rect: tuple[int, int, int, int]) -> str:
    """The filter chain of the OCR decode: ``fps`` frames a second of ``rect`` (x, y, w, h)."""
    x, y, w, h = rect
    return f"fps={fps},crop={w}:{h}:{x}:{y}"


def iter_frames(
    path: Path,
    fps: float,
    rect: tuple[int, int, int, int],
    hw: bool,
    errors: list[str],
    start: float = 0.0,
    dur: float | None = None,
    cancel: threading.Event | None = None,
    exit_codes: list[int] | None = None,
):
    """Yield ``(t, frame)`` for the cropped rectangle; stderr lines go to ``errors``.

    Only ``[start, start + dur]`` is decoded when given; ``t`` is source time. With
    ``cancel`` (a background read), ffmpeg runs out of the console's reach of Ctrl+C and is
    killed once ``cancel`` is set (``decode_raw(isolate=True)``); ``exit_codes`` gets a
    non-zero exit of ffmpeg that the decode did not cause.

    Raises:
        RuntimeError: If ffmpeg produced no frames at all (an
            :class:`~cutpoint.errors.FFmpegError`; the caller may retry
            without hwaccel).
    """
    _x, _y, w, h = rect
    yield from decode_raw(
        path,
        size=(w, h),
        vf=ocr_vf(fps, rect),
        pix_fmt="bgr24",
        start=start if start > 0 else None,
        duration=dur,
        stamp_fps=fps,
        hwaccel="auto" if hw else None,
        fallback_cpu=False,  # the caller switches to the CPU for the remaining segments
        stderr_lines=errors,
        isolate=cancel is not None,
        cancel=cancel,
        exit_codes=exit_codes,
    )


def keep_segments(duration: float, avoid: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """``[0, duration]`` minus the ``avoid`` ranges (each widened by 2 s)."""
    segs = [(0.0, duration)]
    for a, b in sorted(avoid):
        a, b = max(0.0, a - 2.0), b + 2.0
        nxt = []
        for s0, s1 in segs:
            if b <= s0 or a >= s1:
                nxt.append((s0, s1))
                continue
            if a > s0:
                nxt.append((s0, a))
            if b < s1:
                nxt.append((b, s1))
        segs = nxt
    return [(a, b) for a, b in segs if b - a >= 1.0]


def readable_regions(
    regions: tuple[Region, ...], width: int, height: int, name: str
) -> tuple[Region, ...]:
    """The ``regions`` with a crop in a ``width`` x ``height`` frame; the others are left
    out with one WARNING each (a rectangle scaled outside the frame has an empty crop)."""
    out = []
    for r in regions:
        _, _, w, h = scaled(r, width, height)
        if w > 0 and h > 0:
            out.append(r)
        else:
            logger.warning(
                "%s: region %s@%d,%d %dx%d has no pixels in the %dx%d frame; skipped",
                name, r.name, r.x, r.y, r.w, r.h, width, height,
            )  # fmt: skip
    return tuple(out)


def _chunk(
    recs: list[FrameRec],
    regions: tuple[Region, ...],
    state: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> ocr_raw.Chunk:
    """Frame records (and the reader's state after them) as a raw OCR cache chunk."""
    index = {r.name: i for i, r in enumerate(regions)}
    images = {f"{rec.n}-{index[name]}": band for rec in recs for name, band in rec.bands.items()}
    if recs and recs[0].thumb is not None:
        arrays = arrays | {"thumbs": np.stack([rec.thumb for rec in recs])}
    frames = [{"n": rec.n, "t": rec.t, "r": rec.regions} for rec in recs]
    return ocr_raw.Chunk(frames, state, images, arrays)


def _records(chunk: ocr_raw.Chunk, regions: tuple[Region, ...]) -> list[FrameRec]:
    """The frame records of a raw OCR cache chunk."""
    index = {r.name: i for i, r in enumerate(regions)}
    thumbs = chunk.arrays.get("thumbs")
    return [
        FrameRec(
            f["n"],
            f["t"],
            None if thumbs is None else thumbs[i],
            f["r"],
            {
                name: chunk.images[f"{f['n']}-{index[name]}"]
                for name, g in f["r"].items()
                if "rows" in g
            },
        )
        for i, f in enumerate(chunk.frames)
    ]


def _log_score_slots(src_name: str, stats: dict[str, Any]) -> None:
    """One INFO line naming the ``score`` spots of ``stats["score_slots"]`` that were turned
    off as counters, if any (fresh and cached results alike)."""
    off = [k for k, v in stats.get("score_slots", {}).items() if v.get("disabled")]
    if off:
        logger.info(
            "%s: score readout off for %s (a counter, not a score)",
            src_name,
            ", ".join(sorted(off)),
        )


def _raw_region_keys(regions: list[Region]) -> list[dict[str, Any]]:
    """Region keys for :func:`ocr_raw.raw_key`.

    A "score" region is left out of the static-screen thumbnails (:func:`_still_rect`): keyed
    apart, so a read made before with it in them is read again.
    """
    return [r.key() | ({"still": False} if r.kind == "score" else {}) for r in regions]


class _Report:
    """Calls ``on_progress`` with the share (0-1) of the source read at ``t``, at most once
    per percent."""

    def __init__(self, duration: float, on_progress: Callable[[float], None] | None) -> None:
        self.duration = duration
        self.on_progress = on_progress
        self.shown = -1  # the last percent reported

    def __call__(self, t: float) -> None:
        pct = int(100 * min(1.0, t / self.duration)) if self.duration > 0 else 0
        if self.on_progress is not None and pct > self.shown:
            self.shown = pct
            self.on_progress(pct / 100)


def _replay(
    raw: ocr_raw.RawStore,
    meta: dict[str, Any] | None,
    regions: list[Region],
    interp: OcrInterpretation,
    report: _Report,
    cancel: threading.Event | None,
) -> tuple[int, ocr_raw.Chunk | None]:
    """Feed the finished raw chunks of ``meta`` (all of a complete read, or those an
    interrupted one left) to ``interp``; returns how many and the last one."""
    done, last = 0, None
    if meta is None:
        return done, last
    if meta.get("complete"):
        raw.touch()  # in use: another process's cleanup keeps it
    for chunk in raw.chunks(meta):
        if cancel is not None and cancel.is_set():
            raise OcrCancelled
        for rec in _records(chunk, regions):
            interp.add(rec)
            report(rec.t)
        done, last = done + 1, chunk
    return done, last


@dataclass
class _LiveRead:
    """A live OCR read of one source (:func:`_read_segments`): what it reads with, where it
    resumes (``seg``, ``idx``, ``read_errors``: the ``pos`` of the last raw chunk) and where it
    is (``frame_no``, decoder ``hw``, ``failed``, the raw cache ``writer``: None after a disk
    error)."""

    src: Source
    params: OcrParams
    regions: list[Region]
    union: tuple[int, int, int, int]
    reader: OcrReader
    interp: OcrInterpretation
    stats: dict[str, Any]
    report: _Report
    info: dict[str, Any]  # width, height and duration, for the raw read's start
    done: int  # raw chunks replayed before
    writer: ocr_raw.RawStore | None
    debug_dir: Path | None
    cancel: threading.Event | None
    seg: int
    idx: int
    read_errors: int
    frame_no: int
    hw: Any
    failed: bool  # a segment gave no frames: the read stays incomplete
    pending: list[FrameRec] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # the decode errors of this pass
    all_errors: list[str] = field(default_factory=list)  # of the segments read

    def write(self, step: Callable[..., None], *args: Any) -> None:
        """A raw cache write; on a disk error the read goes on without the raw cache."""
        try:
            step(*args)
        except OSError as err:
            logger.warning("%s: raw OCR cache not written: %s", self.src.name, err)
            self.writer = None

    def flush(self, seg: int, idx: int) -> None:
        """Write the pending records as a chunk; the read resumes at frame ``idx`` of
        segment ``seg``. The first flush starts the raw read (an interrupt before it
        leaves no folder)."""
        if self.writer is not None and not self.writer.started:
            self.write(self.writer.start, self.done, self.info)
        if self.writer is None or not self.pending:
            return
        state, arrays = self.reader.state()
        n_errors = sum(bool(READ_ERROR_RE.search(e)) for e in self.all_errors + self.errors)
        at = {"seg": seg, "idx": idx, "frame_no": self.frame_no, "hw": self.hw}
        at |= {"hwaccel": self.stats["hwaccel"], "read_errors": self.read_errors + n_errors}
        at["failed"] = self.failed
        chunk = _chunk(self.pending, self.regions, {"reader": state, "pos": at}, arrays)
        self.write(self.writer.add, chunk)
        self.pending.clear()

    def save_crop(self, out: Path, t: float, r: Region, crop: np.ndarray) -> None:
        """Save the region crop of a new event as a PNG in ``out``."""
        out.mkdir(parents=True, exist_ok=True)
        write_png(out / f"{self.src.path.stem[-24:]}_{t:07.1f}_{r.name}.png", crop)

    def segment(self, si: int, seg_a: float, seg_b: float) -> bool:
        """Read segment ``si`` (``seg_a``-``seg_b``), from where the read resumes; a failed
        GPU decode is retried on the CPU, a failed CPU one marks the read ``failed``. False:
        nothing of it was left to read (its pass did not run)."""
        params, src, cancel = self.params, self.src, self.cancel
        k0 = self.idx if si == self.seg else 0  # frames of the segment already read
        start = seg_a + k0 / params.fps
        if start >= seg_b:
            return False
        by_name = {r.name: r for r in self.regions}
        duration = self.info["duration"]
        while True:
            self.errors.clear()
            exits: list[int] = []  # background: an ffmpeg exit it did not cause
            k = k0
            # "auto" decodes on the CPU when no GPU decoder fits this source: a failed CPU
            # pass is no GPU failure, and a plain retry would only repeat it
            gpu = self.hw and decoder_for(src.io_path) is not None
            try:
                for t, frame in iter_frames(
                    src.io_path,
                    params.fps,
                    self.union,
                    self.hw,
                    self.errors,
                    start,
                    None if seg_b >= duration else seg_b - start,
                    cancel=cancel,
                    exit_codes=exits if cancel is not None else None,
                ):
                    if cancel is not None and cancel.is_set():
                        raise OcrCancelled
                    if k0:  # resumed: the time stamps of an uninterrupted read
                        t = seg_a + k / params.fps
                    k += 1
                    rec, crops = self.reader.read(self.frame_no, t, frame)
                    self.frame_no += 1
                    for name in self.interp.add(rec):
                        if self.debug_dir:
                            self.save_crop(self.debug_dir, t, by_name[name], crops[name])
                    if self.writer is not None:
                        self.pending.append(rec)
                        if len(self.pending) >= ocr_raw.CHUNK_FRAMES:
                            self.flush(si, k)
                    self.report(t)
                if cancel is not None and (cancel.is_set() or exits):
                    raise OcrCancelled  # killed on cancel, or stopped from outside
                self.stats["hwaccel"] = gpu
                return True
            except RuntimeError as err:
                if is_out_of_memory(err) and not isinstance(err, FFmpegError):
                    raise  # an OCR call short of memory is no failed decode (detect_ocr_events)
                if cancel is not None and (cancel.is_set() or exits or gpu):
                    # cancelled or stopped from outside, or a failed GPU pass: the read in
                    # its turn redoes it from the raw chunks (CPU retry, hwaccel stats) as a
                    # serial run would
                    raise OcrCancelled from err
                if k0 and k == k0 and start > seg_b - 1.0:
                    # resumed at the very end of the segment: nothing was left
                    logger.debug("%s: nothing left after %.1fs: %s", src.name, start, err)
                    return True
                if gpu:
                    logger.warning("%s: hwaccel decode failed (%s); retrying on CPU", src.name, err)
                    self.hw = False
                    continue
                logger.warning("%s: no frames in %.0f-%.0fs: %s", src.name, seg_a, seg_b, err)
                self.failed = True
                return True


def _read_segments(read: _LiveRead, segments: list[tuple[float, float]], tries: int) -> None:
    """Read the ``segments`` from where ``read`` resumes, then finish its raw cache (complete,
    or left incomplete when a part gave no frames: ``stats["incomplete"]`` counts the
    ``tries``)."""
    stats, name = read.stats, read.src.name
    for si, (seg_a, seg_b) in enumerate(segments):
        if si < read.seg:
            continue
        if read.segment(si, seg_a, seg_b):
            read.all_errors += read.errors
    read.errors.clear()
    read_errors = [e for e in read.all_errors if READ_ERROR_RE.search(e)]
    if read_errors:
        logger.warning(
            "%s: %d read errors during OCR decode (e.g. %s)",
            name,
            len(read_errors),
            read_errors[0],
        )
    stats["read_errors"] = read.read_errors + len(read_errors)
    if read.failed:
        stats["incomplete"] = tries + 1
    read.flush(len(segments), 0)
    if read.writer is not None and read.failed:
        logger.info("%s: raw OCR cache left incomplete (a part gave no frames)", name)
        read.write(read.writer.fail)
    elif read.writer is not None:
        read.write(read.writer.finish, {k: stats[k] for k in ("hwaccel", "read_errors")})


def _save(
    src: Source,
    cpath: Path,
    key: dict[str, Any],
    stats: dict[str, Any],
    interp: OcrInterpretation,
    t_start: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The events ``interp`` saw in the read (or replay), logged and cached under ``cpath``;
    returns (events, stats)."""
    stats["elapsed_s"] = round(time.perf_counter() - t_start, 2)
    events = interp.finish()
    if stats.get("replayed"):
        logger.info(
            "%s: %d OCR events, %d frames, replayed from the raw OCR cache: 0 live OCR calls "
            "(the cached read made %s), %.1fs",
            src.name,
            len(events),
            stats["frames"],
            stats["ocr_calls"],
            stats["elapsed_s"],
        )
    else:
        logger.info(
            "%s: %d OCR events, %d frames, OCR calls %s, %.1fs",
            src.name,
            len(events),
            stats["frames"],
            stats["ocr_calls"],
            stats["elapsed_s"],
        )
    if "det_skipped" in stats:
        skipped = sum(stats["det_skipped"].values())
        logger.info(
            "%s: det gate skipped on %d of %d region reads (unchanged since its last det call)",
            src.name,
            skipped,
            skipped + sum(stats["det_calls"].values()),
        )
    _log_score_slots(src.name, stats)
    cache.save_json(
        cpath, {"source": str(src.path), "params": key, "stats": stats, "events": events}
    )
    return events, stats


def detect_ocr_events(
    src: Source,
    params: OcrParams,
    engine: OcrEngine | None,
    force: bool = False,
    debug_dir: Path | None = None,
    avoid: list[tuple[float, float]] | None = None,
    on_progress: Callable[[float], None] | None = None,
    make_engine: Callable[[], OcrEngine] | None = None,
    cancel: threading.Event | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OCR HUD events for one source (cached per source + parameters).

    What menu and scoreboard frames show is dropped (:func:`menu_rules`): every frame records the text boxes the "det" gates
    last detected in its regions other than feeds (a region keeps its boxes on
    the frames it is skipped) and the still run of the decoded HUD area without the
    "score" regions (:func:`_still_rect`, :func:`static_frames`; when that area is at least :attr:`MenuParams.static_min_area` of
    the frame), and after the pass the sightings in menu frames
    are forgotten.
    No OCR runs for this; ``stats["menu_frames"]`` counts them (``stats["menu_spans"]``:
    their ``[start, end]`` spans, :func:`~cutpoint.events.menu_spans.merge_menu_frames`), of which
    ``stats["static_frames"]`` held still and ``stats["text_frames"]`` showed many
    lines of new text that stay (:func:`text_frames`).

    The read (:class:`OcrReader`) and its interpretation (:class:`OcrInterpretation`)
    are apart: the reader's records go into a raw OCR cache
    (:mod:`cutpoint.events.ocr_raw`), so when only the interpretation changed (a
    :data:`GENERIC_VERSION` bump) the result is replayed from it in seconds, with
    no OCR call and no ``engine`` (``stats["replayed"]`` is then True; ``ocr_calls``
    are those of the cached read), and an interrupted read resumes after its last
    finished chunk.

    An out-of-memory error of an OCR call (:func:`~cutpoint.errors.is_out_of_memory`, not a
    failed decode) is raised as it is, also with ``cancel``: it is not retried here. The
    background reader (:mod:`cutpoint.events.ocr_prefetch`) hands such a read back to its
    turn; inline it stops the run.

    Args:
        src: Source video (read once, sequentially).
        params: OCR parameters.
        engine: Loaded OCR engine, or None to only read the caches.
        force: Ignore the cache and drop the raw OCR cache (read the footage again).
        debug_dir: If set, save the region crop of every new event as a PNG (the
            footage is always read then).
        avoid: Time ranges never decoded (bad sectors); part of the cache key.
        on_progress: Called with the share (0-1) of the source decoded so far,
            at most once per percent.
        make_engine: Instead of ``engine``: makes the engine only when the footage is
            read (a replay or a cache hit loads no OCR model).
        cancel: For a read in a background thread (:mod:`cutpoint.events.ocr_prefetch`):
            once set, the read stops at the next frame with :class:`OcrCancelled`, its
            ffmpeg killed and nothing committed after the last finished raw chunk. ffmpeg
            then runs isolated from Ctrl+C and is killed at once on ``cancel``. A pass
            that ffmpeg left with an exit the decode did not cause (the console's
            Ctrl+Break reaches every process group) and a failed GPU pass also raise
            :class:`OcrCancelled`, without committing the pass: the caller reads the
            source itself then, resuming from the raw chunks, as a serial run would.

    Returns:
        (events, stats).

    Raises:
        OcrCancelled: With ``cancel``: when it was set, ffmpeg was stopped from outside or a
            GPU pass failed.
    """
    game = game_key_for_path(str(src.path), params.game, params.project)
    preset = load_preset(game, params.project)
    regions = preset.all_regions() if preset else ()
    if not regions:
        if preset is not None:
            logger.info("%s: HUD preset %r has no usable region; skipping OCR", src.name, game)
        else:
            logger.info(
                "%s: no HUD preset for %r; skipping OCR (run with --discover-regions)",
                src.name,
                game,
            )
        return [], {"skipped": "no preset", "game": game}
    avoid = sorted(avoid or [])
    newest = str(preset.feed.get("newest", "bottom")) if preset is not None else "bottom"
    raw = ocr_raw.RawStore(
        src.cdir,
        ocr_raw.raw_key(
            params.fps,
            _raw_region_keys(regions),
            newest,
            MIN_TEXT_SCORE,
            avoid,
            [DET_SKIP_SCALE, DET_SKIP_TILE, DET_SKIP_MAX_DIFF] if params.det_skip else None,
        ),
    )
    key = params.key(game, preset) | ({"avoid": avoid} if avoid else {})
    key["raw"] = cache.params_hash(raw.key)
    cpath = layer_cache_path(src.cdir, "ocr", key)
    can_read = engine is not None or make_engine is not None
    tries = 0  # reads of this source that left a part unread
    if not force and (hit := cache.load_json(cpath)) is not None:
        tries = int(hit.get("stats", {}).get("incomplete", 0))
        # a read that left a part unread is read again, but only once more: a part that
        # fails twice is taken as damaged
        if not (tries and tries < MAX_INCOMPLETE_READS and can_read):
            _log_score_slots(src.name, hit.get("stats", {}))
            return hit["events"], hit.get("stats", {})
    if force and can_read:  # without an engine nothing would read it again
        raw.drop()
    meta = None if debug_dir else raw.meta()
    if meta is not None and meta.get("failed"):
        meta = None  # a read with a part that gave no frames: read it all again
    if not can_read and not (meta and meta.get("complete")):
        return [], {"skipped": "not cached"}

    if meta is not None:
        width, height, duration = (meta["info"][k] for k in ("width", "height", "duration"))
    else:
        info = probe(src.io_path)
        width, height, duration = info.width, info.height, info.duration
    regions = readable_regions(regions, width, height, src.name)
    if not regions:
        return [], {"skipped": "no region in the frame", "game": game}
    rects = {r.name: scaled(r, width, height) for r in regions}
    union = _union(list(rects.values()))
    stats: dict[str, Any] = {"game": game, "hwaccel": False}
    if preset is not None and preset.path is not None:
        stats["preset"] = str(preset.path)
    interp = OcrInterpretation(src.name, params, preset, regions, (width, height), stats)
    t_start = time.perf_counter()
    segments = keep_segments(duration, avoid)
    report = _Report(duration, on_progress)
    # replay the finished raw chunks: all of a complete read, or those an interrupted one left
    done, last = _replay(raw, meta, regions, interp, report, cancel)
    if meta is not None and meta.get("complete") and done == meta["chunks"]:
        raw.touch()
        stats |= meta.get("stats", {})  # hwaccel and read errors of the read
        stats["replayed"] = True
    elif not can_read:  # a damaged raw read
        return [], {"skipped": "not cached"}
    else:
        if engine is None:
            engine = make_engine()
        reader = OcrReader(
            regions, rects, union, engine, params, height / REF_H, newest, (width, height)
        )
        # where the read goes on: segment, frame in it, frame number, decoder, read errors
        pos: dict[str, Any] = {
            "seg": 0,
            "idx": 0,
            "frame_no": 0,
            "hw": params.hwaccel,
            "hwaccel": False,
            "read_errors": 0,
        }
        if last is not None:
            reader.restore(last.state["reader"], last.arrays)
            pos = last.state["pos"]
            stats["hwaccel"] = pos["hwaccel"]
            logger.info(
                "%s: resuming the OCR read after %d frames (raw OCR cache)",
                src.name,
                stats["frames"],
            )
        read = _LiveRead(
            src=src,
            params=params,
            regions=regions,
            union=union,
            reader=reader,
            interp=interp,
            stats=stats,
            report=report,
            info={"width": width, "height": height, "duration": duration},
            done=done,
            writer=raw if raw.writable() else None,
            debug_dir=debug_dir,
            cancel=cancel,
            seg=pos["seg"],
            idx=pos["idx"],
            read_errors=pos["read_errors"],
            frame_no=pos["frame_no"],
            hw=pos["hw"],
            failed=bool(pos.get("failed")),
        )
        _read_segments(read, segments, tries)
    return _save(src, cpath, key, stats, interp, t_start)
