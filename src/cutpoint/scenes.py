"""Scene cuts of a source for :mod:`cutpoint.analyze`: PySceneDetect on a small proxy.

The full pass (:func:`scene_cuts`, cached as :data:`SCENES_FILE`), its decode split between
NVDEC and the CPU (#730), the overlap with Whisper (:class:`_SceneJob`, #725), and the windows
around candidate edges of a style that does not weigh the scene curve (:class:`EdgeCuts`).
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from cutpoint import cache, ffmpeg_utils, intervals, paths
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import (
    NVDEC_DOWNLOAD,
    READ_ERROR_RE,
    STOP_GRACE_S,
    HwDecoder,
    ProgressResult,
    RunCancelled,
    decoder_for,
    first_frame_time,
    gpu_chain,
    nvdec_download,
    probe,
    run_progress,
)
from cutpoint.styles import Style
from cutpoint.utils import clock

logger = logging.getLogger(__name__)

LONG_FILE_S = 1800  # above this, scene detection decodes keyframes only
SCENE_OVERLAP_ENV = "ANALYZE_SCENE_OVERLAP"  # CUTPOINT_ANALYZE_SCENE_OVERLAP=1: opt in (#725)
SCENE_JOIN_S = 8.0  # a cancelled scene thread gets this long to stop (a daemon, never waited on)
SCENE_POLL_S = 0.5  # waits on the scene thread in steps this long: Ctrl+C gets through on Windows
# The full scene pass's cuts per part cache folder: "-v3" since the proxy is lossless (#994,
# a lossy one added cuts at its keyframes); "-v2" counted every pass from the container start
# (#976). An older file is not read, the source's scenes are detected again.
SCENES_FILE = "scenes-v3.json"
# Scene cuts of a style that does not weigh them, only around candidate edges (#730, EdgeCuts)
SCENE_WINDOWS_FILE = "scenes-windows.json"  # per part cache folder, next to SCENES_FILE
SCENE_WINDOWS_VERSION = 3  # 3: lossless proxy (#994), 2: container clock (#976); else decoded again
SNAP_REACH_S = 1.0  # _snap moves an edge to a scene cut at most this far
SCENE_WARMUP_S = 1.5  # picture the detector needs around a cut (2-frame window, 0.5 s min)
WINDOW_PAD_S = SNAP_REACH_S + SCENE_WARMUP_S  # a window reaches this far either side of an edge
SCENE_GRID_S = 0.1  # the full proxy's frame step (10 fps, _proxy_args): windows start on it
# A recovered part of a container that seeks inexactly (MPEG-TS) is decoded from this far before
# its start (_lead_in): ffmpeg's seek there lands on the next keyframe, past cuts near the start
LEAD_IN_S = 10.0
# ... but never from before the end of the unreadable stretch before it (the read-error record's
# ``unreadable_until``), and not at all when less than this is left: about one GOP of a common
# recording (OBS and stream encoders put a keyframe every 2 s), so a shorter lead-in mostly lands
# on the keyframe the plain seek does and only costs its measuring decode
LEAD_IN_MIN_S = 2.0


def scene_overlap() -> bool:
    """True when ``CUTPOINT_ANALYZE_SCENE_OVERLAP=1``: a source's scene cuts are detected
    while Whisper transcribes it."""
    return (paths.env(SCENE_OVERLAP_ENV) or "").strip() == "1"


def _first_shift(src: Path, offset: float, duration: float) -> float | None:
    """Where on the container clock (from :attr:`MediaInfo.fstart`) the first frame a whole
    scene pass of ``src`` decodes lies (:func:`_proxy_args`), or None when ffmpeg's own count
    is that clock: a pass that seeks (in a container that does not seek exactly a recovered
    part places its frames by :func:`_lead_in` instead), a whole pass of a container that seeks
    exactly, and a source ffprobe or the one-frame decode fails on (the pass then counts as
    ffmpeg does)."""
    if offset > 0 or _seeks_exactly(src):
        return None
    try:
        first = first_frame_time(src, keyframes=duration > LONG_FILE_S)
        return first - probe(src).fstart
    except FFmpegError as exc:
        logger.debug("%s: no first frame time (%s); the scene pass counts as ffmpeg does", src, exc)
        return None


def _lead_in(
    src: Path, offset: float, duration: float, readable_from: float | None
) -> tuple[float, float] | None:
    """The seek point and the shift of a scene pass of ``src`` from ``offset`` (a recovered
    part) in a container that does not seek exactly (:func:`_seeks_exactly`), or None: a pass
    from the start, a container that seeks exactly, no ``readable_from`` known (a read-error
    record older than it), less than :data:`LEAD_IN_MIN_S` between it and ``offset``, and a
    source ffprobe or the one-frame decode fails on (the pass then seeks ``offset`` as ffmpeg
    does).

    MPEG-TS seeks to the next keyframe after ``offset`` (``-ss 6.4`` decodes from 8.02), so cuts
    before it were lost. The pass seeks :data:`LEAD_IN_S` earlier instead, but never before
    ``readable_from`` (the end of the unreadable stretch before the part,
    :func:`~cutpoint.analyze.find_resume_point`: the read error must not come back), and the first
    frame decoded from there is put at its container time less ``offset`` (measured by a decode of
    its own: the seek may land anywhere, or clamp to the start), so ``fps`` drops the lead-in and
    the proxy starts at ``offset`` (:func:`_proxy_args`)."""
    if offset <= 0 or readable_from is None or _seeks_exactly(src):
        return None
    seek = round(max(0.0, offset - LEAD_IN_S, readable_from), 3)
    if offset - seek < LEAD_IN_MIN_S:
        return None
    try:
        first = first_frame_time(src, keyframes=duration > LONG_FILE_S, start=seek)
        return seek, first - probe(src).fstart - offset
    except FFmpegError as exc:
        logger.debug("%s: no first frame time (%s); the scene pass seeks as ffmpeg does", src, exc)
        return None


REINIT_OFF = ["-reinit_filter", "0"]  # input option: one filter graph per pass (_proxy_args)


def _proxy_args(
    duration: float,
    offset: float = 0.0,
    limit: float | None = None,
    shift: float | None = None,
    lead_in: tuple[float, float] | None = None,
) -> tuple[list[str], list[str], str, list[str]]:
    """The scene proxy's input options, output options, filter chain and encoder arguments.

    Every pass counts from the container start (#976), so its cuts are on the clock clips are
    cut on: ``fps`` with ``start_time=0`` makes the proxy's first frame the seek point (the
    container start without one) and pads a video that starts later with its first frame.
    That is ffmpeg's own count in a container that seeks exactly (:func:`_seeks_exactly`). In
    MPEG-TS ffmpeg 6.1 counts a whole pass from the video instead (``-an``), so the first
    decoded frame is put at ``shift`` (:func:`_first_shift`: its container time, also when the
    file starts in the middle of a GOP); ffmpeg's own timestamps stay, so a timestamp reset
    (joined TS files) still runs on. A seek is exact only in a container that seeks exactly;
    elsewhere ``lead_in`` (:func:`_lead_in`: seek point and shift) seeks earlier and puts the
    first decoded frame at ``shift``, so ``fps`` drops the frames before ``offset``.

    ``-reinit_filter 0`` keeps one filter graph for the whole pass: where the video parameters
    change mid-file (a joined file's second half with other color tags) ffmpeg 9 would rebuild
    it, and ``setpts`` re-reads its start and ``fps`` pads from 0 again, so the timestamps
    after the join are wrong or lost."""
    long_file = duration > LONG_FILE_S
    pre = [*REINIT_OFF, *(["-skip_frame", "nokey"] if long_file else [])]
    if lead_in is not None:
        if lead_in[0] > 0:
            pre += ["-ss", f"{lead_in[0]:.3f}"]
    elif offset > 0:
        pre += ["-ss", f"{offset:.3f}"]
    post = ["-t", f"{limit:.3f}"] if limit else []
    rate, width = (2, 256) if long_file else (10, 320)
    vf = f"fps={rate}:start_time=0,scale={width}:-2"
    if lead_in is not None:
        vf = f"setpts=PTS-STARTPTS+{lead_in[1]:.6f}/TB,{vf}"
    elif shift is not None and offset <= 0:
        vf = f"setpts=PTS-STARTPTS+{shift:.6f}/TB,{vf}"
    # Lossless: a lossy proxy has a keyframe every 250 frames (25 s at 10 fps) that adds to the
    # detector's content value, so cuts appeared there and moved with the pass's start (#994)
    return pre, post, vf, ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]


def _fixed_scale(vf: str, src: Path) -> str:
    """``vf`` with the proxy's ``scale=W:-2`` given the height of ``src``'s probed picture
    (``scale=W:H``, even): with ``-reinit_filter 0`` a picture that changes size mid-file
    would reach libx264 in a new size, which crashes ffmpeg 9 now and then. ``vf`` as it is
    when the probe fails or has no size."""
    head, sep, rest = vf.rpartition(",scale=")
    if not sep or not rest.endswith(":-2"):
        return vf
    try:
        info = probe(src)
    except FFmpegError:
        return vf
    if info.width <= 0 or info.height <= 0:
        return vf
    width = int(rest[: -len(":-2")])
    return f"{head}{sep}{width}:{max(2, round(width * info.height / info.width / 2) * 2)}"


def _scene_decode(
    hw: HwDecoder | None, vf: str, src: Path, download: bool = False
) -> tuple[list[str], str]:
    """The decoder's input options and the filter chain of the scene proxy. On a GPU decoder and
    an 8- or 10-bit 4:2:0 source the frames stay on the GPU until the ``fps`` filter
    dropped the ones the detector does not read, so only those are downloaded. Other formats
    (4:4:4, 12-bit, unknown) and the CPU keep the plain path. The proxy's size is fixed
    (:func:`_fixed_scale`).

    A container that does not seek exactly (MPEG-TS, possibly joined files), and any source with
    ``download`` (its stream parameters change mid-file, :func:`_run_scene_proxy`), is decoded
    without a ``hwdownload`` in the chain: NVDEC reinitialises at a join and ffmpeg rebuilds the
    graph despite ``-reinit_filter 0`` (``setpts`` and ``fps`` start again, the encoder drops
    the rest, exit 0). CUDA then downloads the frames itself (``-hwaccel cuda`` alone, the plain
    chain); VAAPI always needs the download, so it decodes on the CPU."""
    vf = _fixed_scale(vf, src)
    if hw is None:
        return [], vf
    if download or not _seeks_exactly(src):
        if hw.kind == "cuda":
            try:
                info = probe(src)
            except FFmpegError:
                info = None  # the decode reports an unreadable file itself
            if info is not None and nvdec_download(info.vcodec, info.pix_fmt or "") is None:
                logger.debug(
                    "%s: %s %s is outside what NVDEC decodes (8/10-bit 4:2:0 HEVC, VP9, AV1, "
                    "8-bit 4:2:0 H.264); NVDEC may not decode it and ffmpeg then decodes it "
                    "in software",
                    src.name, info.vcodec or "unknown", info.pix_fmt or "unknown",
                )  # fmt: skip
            return hw.input_args(), vf
        return [], vf
    info = probe(src)
    if hw.kind == "cuda":
        fmt = nvdec_download(info.vcodec, info.pix_fmt or "")
    else:
        fmt = NVDEC_DOWNLOAD.get(info.pix_fmt or "")
    if fmt is None:
        return hw.input_args(), gpu_chain(hw, vf) or vf
    head, sep, rest = vf.partition(",scale=")  # the proxy chain's fps filter comes before it
    return hw.input_args(keep_on_gpu=True), f"{head},hwdownload,format={fmt}{sep}{rest}"


def _run_scene_proxy(
    command: Callable[[bool], list[str]],
    proxy: Path,
    seconds: float,
    label: str,
    cancel: threading.Event | None = None,
) -> ProgressResult:
    """Run the scene proxy's ffmpeg ``command(False)``; when its frames stayed on the GPU
    (``hwdownload``, :func:`_scene_decode`) and ffmpeg dropped frames, run ``command(True)``
    again, which does not keep them there. Returns the last run's result.

    The stream parameters changed mid-file (#1031: an MP4 or MKV whose second part has another
    size): the GPU decoder starts again with new frames, and ffmpeg rebuilds the filter graph
    for them despite ``-reinit_filter 0``. The new ``fps`` pads from 0 again, the encoder drops
    those frames but keeps one, and every later cut is a frame late (or lost). Frames that
    are not on the GPU keep one graph, as on the CPU. A pass whose ``fps`` output is the only
    frame source drops none, so a normal file runs once."""
    opts = {"cancel": cancel} if cancel is not None else {}  # plain call when not overlapped
    first = command(False)
    res = run_progress(first, seconds, label, **opts)
    if not any("hwdownload" in arg for arg in first) or not res.dropped:
        return res
    logger.info(
        "%s: the video changes its size or format mid-file and the GPU decode lost its frame "
        "timing there (%d frames dropped); making it again without keeping its frames on the GPU",
        label,
        res.dropped,
    )
    proxy.unlink(missing_ok=True)  # the re-run starts clean
    return run_progress(command(True), seconds, label, **opts)


def _detect(proxy: Path, path: Path, cancel: threading.Event | None = None) -> list[float]:
    """Scene cuts of ``proxy`` (deleted after), saved to ``path``.

    With ``cancel`` (a :class:`_SceneJob`), setting it stops the detection within about 0.1 s plus a
    frame decode and raises :class:`~cutpoint.ffmpeg_utils.RunCancelled`, writing nothing."""
    try:
        cuts = _scan(proxy, cancel)  # its video handle is closed on return
    finally:
        try:
            proxy.unlink(missing_ok=True)
        except OSError as exc:  # never replace the original exception
            logger.debug("could not delete %s: %s", proxy, exc)
    cache.save_json(path, cuts)
    return cuts


def _scan(proxy: Path, cancel: threading.Event | None) -> list[float]:
    from scenedetect import AdaptiveDetector, SceneManager, open_video  # noqa: PLC0415

    video = open_video(str(proxy))
    manager = SceneManager()
    manager.auto_downscale = False
    manager.add_detector(AdaptiveDetector(min_scene_len=5))

    done = threading.Event()

    def watch() -> None:
        # PySceneDetect calls a callback only at a cut, so a stop is requested from here. It is
        # repeated: detect_scenes clears its stop flag when it starts, so one early stop can be lost.
        while not done.is_set():
            if cancel.wait(0.1):  # type: ignore[union-attr]
                manager.stop()
                done.wait(0.1)

    watcher = None
    if cancel is not None:
        watcher = threading.Thread(target=watch, name="scene-cancel", daemon=True)
        watcher.start()
    try:
        manager.detect_scenes(video, show_progress=False)
    finally:
        done.set()
        if watcher is not None:
            watcher.join(1.0)
        # The OpenCV backend has no close(); a traceback keeps the handle (and the file open).
        release = getattr(getattr(video, "capture", None), "release", None)
        if release is not None:
            release()
    if cancel is not None and cancel.is_set():
        raise RunCancelled(f"scene detection {proxy.parent.name}")
    return [round(s[0].seconds, 3) for s in manager.get_scene_list()[1:]]


def scene_cuts(
    src: Path,
    cdir: Path,
    duration: float,
    offset: float = 0.0,
    limit: float | None = None,
    cancel: threading.Event | None = None,
    cpu_free: bool = False,
    readable_from: float | None = None,
) -> list[float]:
    """Scene-cut times (s) via PySceneDetect AdaptiveDetector on a tiny proxy.

    Normal files: 320px / 10 fps proxy. Files longer than ``LONG_FILE_S``:
    keyframes-only decode (``-skip_frame nokey``) at 256px / 2 fps, which is
    roughly I/O-bound even for multi-hour VODs (cut precision ~ one GOP).

    Args:
        src: Source video.
        cdir: Cache folder of this source (or recovered part).
        duration: Length to analyse (s).
        offset: Seek here first; returned times are relative to it.
        limit: Stop decoding after this many seconds (stays clear of a known
            unreadable region instead of hitting it again).
        cancel: For a run in a background thread (:class:`_SceneJob`): once set, ffmpeg is
            killed or the detection stopped, the proxy removed and :class:`~cutpoint.ffmpeg_utils.RunCancelled` raised,
            before any :data:`SCENES_FILE` is written.
        cpu_free: The caller knows no CPU-heavy step (the background HUD OCR, Whisper on the
            CPU) runs meanwhile, so the decode may be split between NVDEC and the CPU
            (:func:`_hybrid_plan`, #730). False (the default): one decode, as before.
        readable_from: For a recovered part (``offset`` above 0): where the unreadable stretch
            before it ends (absolute s, the read-error record's ``unreadable_until``). In a
            container that does not seek exactly the pass starts a little before ``offset``,
            but never before this (:func:`_lead_in`); None: it seeks ``offset``.
    """
    path = cdir / SCENES_FILE
    cached = cache.load_json(path)
    if cached is not None:
        return cached
    logger.info("%s: detecting scene cuts", src.name)
    importlib.import_module("scenedetect")  # not installed: fails before any decode
    dopts = {"cancel": cancel} if cancel is not None else {}  # plain call when not overlapped
    plan = _hybrid_plan(src, duration, offset, limit, cpu_free)
    if plan is not None and (cuts := _hybrid_cuts(src, cdir, duration, *plan, cancel)) is not None:
        cache.save_json(path, cuts)
        return cuts
    fixed = cdir / "proxy.mp4"
    ffmpeg_utils.sweep_tmp(fixed)  # leftovers of killed runs; other live runs keep theirs
    with contextlib.suppress(OSError):
        fixed.unlink(missing_ok=True)  # the fixed name of older versions
    proxy = ffmpeg_utils.unique_tmp(fixed)  # another run may share cdir
    gpu = scene_decode_mode() != "cpu"
    lead = _lead_in(src, offset, duration, readable_from)
    args = _proxy_args(duration, offset, limit, _first_shift(src, offset, duration), lead)
    try:
        res = _make_proxy(src, proxy, args, duration, cancel, gpu=gpu)
        # ffmpeg stops at a read error and exits 0: one in the lead-in left no frame past the
        # part's start (out_time is part-relative, fps drops the lead-in); one later in the part
        # leaves the proxy as the plain seek would, so it is kept
        failed = (
            lead is not None
            and bool(READ_ERROR_RE.search(res.stderr))
            and res.out_time < SCENE_GRID_S
        )
    except FFmpegError:
        if lead is None:
            raise
        failed = True
    if failed:  # the lead-in reached back to the damage before a recovered part
        logger.info(
            "%s: the scene decode's lead-in before %s hit a read error; decoding from there",
            src.name,
            clock(offset),
        )
        proxy.unlink(missing_ok=True)
        _make_proxy(src, proxy, _proxy_args(duration, offset, limit), duration, cancel, gpu=gpu)
    return _detect(proxy, path, **dopts)


# The scene pass split between NVDEC and the CPU (#730 PR 2). Measured on an RTX 3080 and an
# 8-core i7-9700K (1080p60 H.264, 600 s): NVDEC alone 57 s, the CPU alone 74 s, both at once
# 38 s with the same cuts (P/R 0.98-1.00 against the single decode), at the cost of every core.
SCENE_DECODE_ENV = "SCENE_DECODE"  # CUTPOINT_SCENE_DECODE=auto (default) | hybrid | gpu | cpu
SCENE_DECODE_MODES = ("auto", "hybrid", "gpu", "cpu")
# Below this a source decodes on NVDEC in about 10 s (1080p60, about 685 frames/s): a split
# saves 3 s or less, not worth a second ffmpeg start, seek and scan plus every core.
HYBRID_MIN_S = 120.0
HYBRID_HEADROOM_CPUS = 2  # left to the NVDEC half's ffmpeg (download, scale, x264) and the scans
HYBRID_MIN_THREADS = 4  # fewer CPU decode threads give too little to pay for the split
# Codecs the CPU decodes fast enough next to NVDEC: H.264 1080p60 and 4K60 halve or better the
# pass; 4K120 HEVC decodes about 3x slower per CPU thread, so its CPU half took twice as long
# as NVDEC alone (113 s vs 57 s for 218 s).
HYBRID_CODECS = frozenset({"h264"})
# 8-bit 4:2:0 only: NVDEC here decodes no 10-bit or 4:2:2 H.264 (the NVDEC half fails, or decodes
# on the CPU without saying so)
HYBRID_PIX_FMTS = frozenset({"yuv420p", "yuvj420p"})
# Containers whose input seek is frame-accurate (WebM too, also without cues, #1038); MPEG-TS
# seeks inexactly and moves the CPU half's cuts (up to 0.6 s, or loses them)
HYBRID_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv", ".webm"})
# The CPU decodes at about this share of NVDEC's rate per decode thread (74 s on 8 threads vs
# 57 s); the split puts each half's end at about the same time: 8 threads 0.57, 6 0.63.
CPU_RATE_PER_THREAD = 0.096
# Each half decodes this far past the split. The CPU half's detector gives no cut in its first
# 0.6 s (a frame score needs the frame before, min_scene_len 5 frames after the first scored
# frame), and must see a cut up to 0.5 s (min_scene_len) before the split to drop one just
# after it as one pass would: 1.1 s needed, 3 s leaves room for a chain of close cuts. The
# NVDEC half needs 2 frames (the detector's window) past the split to decide a cut before it.
SEAM_S = 3.0
# One pass never gives two cuts closer than min_scene_len (5 frames at 10 fps = 0.5 s): a CPU
# cut this close after the last NVDEC cut is the same cut, a frame off at the seam.
SEAM_DEDUP_S = 0.45


def scene_decode_mode() -> str:
    """``CUTPOINT_SCENE_DECODE``: ``auto`` (default, and any unknown value), ``hybrid``,
    ``gpu`` or ``cpu``."""
    mode = (paths.env(SCENE_DECODE_ENV) or "").strip().lower()
    return mode if mode in SCENE_DECODE_MODES else "auto"


def _hybrid_threads() -> int:
    """CPU decode threads of the split's CPU half: every logical CPU but the headroom."""
    return (os.cpu_count() or 1) - HYBRID_HEADROOM_CPUS


def _hybrid_split(duration: float, threads: int) -> float:
    """Where the NVDEC half ends and the CPU half takes over, on the proxy's frame grid
    (:data:`SCENE_GRID_S`), so both halves' frames are those of one pass: the share of NVDEC
    is its rate over both rates (:data:`CPU_RATE_PER_THREAD`)."""
    share = 1.0 / (1.0 + CPU_RATE_PER_THREAD * threads)
    return round(round(duration * share / SCENE_GRID_S) * SCENE_GRID_S, 3)


def _hybrid_plan(
    src: Path, duration: float, offset: float, limit: float | None, cpu_free: bool
) -> tuple[HwDecoder, float, int] | None:
    """The NVDEC decoder, split point and CPU threads of a split scene pass of ``src``, or None
    for one decode. ``CUTPOINT_SCENE_DECODE``: ``gpu`` and ``cpu`` never split; ``hybrid``
    splits whenever NVDEC decodes ``src`` and its container seeks exactly, as the CPU half
    seeks (:func:`_seeks_exactly`); ``auto`` only when also ``cpu_free`` (no CPU-heavy step
    runs meanwhile), the source is at least :data:`HYBRID_MIN_S` long, in one of
    :data:`HYBRID_CODECS` and :data:`HYBRID_PIX_FMTS`, and :data:`HYBRID_MIN_THREADS` CPU
    threads are left after the headroom.
    Only a whole, clean source of up to :data:`LONG_FILE_S` (a longer one decodes keyframes
    only)."""
    mode = scene_decode_mode()
    if mode in ("gpu", "cpu") or offset or limit or duration > LONG_FILE_S:
        return None
    if not _seeks_exactly(src):
        return None
    threads = _hybrid_threads()
    if mode == "auto" and (
        not cpu_free
        or duration < HYBRID_MIN_S
        or threads < HYBRID_MIN_THREADS
        or not _hybrid_source_ok(src)
    ):
        return None
    hw = decoder_for(src)
    if hw is None or hw.kind != "cuda":  # VAAPI or the CPU: one decode, as before
        return None
    threads = max(1, threads)
    return hw, _hybrid_split(duration, threads), threads


def _hybrid_source_ok(src: Path) -> bool:
    """``src`` is a source ``auto`` splits: see :func:`_hybrid_plan`."""
    info = probe(src)
    return info.vcodec in HYBRID_CODECS and info.pix_fmt in HYBRID_PIX_FMTS


def _seeks_exactly(src: Path) -> bool:
    """``src`` is in a container whose input seek lands on the frame asked for
    (:data:`HYBRID_SUFFIXES`): a scene pass may start in the middle (a split pass's CPU half,
    an edge window). MPEG-TS (#976: cuts 1.4 s late on ffmpeg 6.1) and the others decode from
    the container start."""
    return src.suffix.lower() in HYBRID_SUFFIXES


def _merge_seam(
    gpu: Sequence[float], cpu: Sequence[float], split: float, cstart: float
) -> list[float]:
    """The cuts of a split pass: the NVDEC half's before ``split``, the CPU half's (times from
    its start ``cstart``) from ``split`` on; a CPU cut closer than :data:`SEAM_DEDUP_S` after
    the last NVDEC cut is that cut seen a frame off and dropped."""
    eps = 1e-6
    out = [round(c, 3) for c in gpu if c < split - eps]
    for x in cpu:
        t = round(cstart + x, 3)
        if t < split - eps or (out and t - out[-1] < SEAM_DEDUP_S):
            continue
        out.append(t)
    return out


def _hybrid_cuts(
    src: Path,
    cdir: Path,
    duration: float,
    hw: HwDecoder | None,
    split: float,
    threads: int,
    cancel: threading.Event | None = None,
) -> list[float] | None:
    """Scene cuts of ``src`` from two proxies made and scanned at once: NVDEC decodes
    ``[0, split + SEAM_S]``, the CPU (``threads`` decode threads) ``[split - SEAM_S, end]``;
    merged by :func:`_merge_seam` (``hw`` None decodes both halves on the CPU). None when a
    half failed (a WARNING; the caller decodes in one pass). Both proxies are removed, also on
    Ctrl+C or ``cancel`` (then :class:`~cutpoint.ffmpeg_utils.RunCancelled`)."""
    cstart = round(max(0.0, split - SEAM_S), 3)
    gend = min(duration, split + SEAM_S)
    # a container that seeks exactly (_hybrid_plan): both halves count as ffmpeg does
    _pre, _post, vf, enc = _proxy_args(duration)
    tag = f"{os.getpid()}-{threading.get_ident()}"  # another run may share the cache folder
    dec_args, chain = _scene_decode(hw, vf, src)
    # the CPU half converts to the format NVDEC downloads before it scales, as that half does,
    # so the same frames give the same proxy pixels
    cpu_vf = (
        chain.replace("hwdownload,", "", 1) if "hwdownload," in chain else _fixed_scale(vf, src)
    )
    out = ["-an", "-vf"]

    def gpu_args(download: bool) -> list[str]:  # download: see _run_scene_proxy
        dec, gchain = _scene_decode(hw, vf, src, True) if download else (dec_args, chain)
        return [*REINIT_OFF, *dec, "-i", str(src), "-t", f"{gend:.3f}", *out, gchain]

    halves = {
        "gpu": (cdir / f"proxy-gpu-{tag}.mp4", gpu_args(False), gend),
        "cpu": (
            cdir / f"proxy-cpu-{tag}.mp4",
            # both halves count from the container start (the chain's fps start_time, #976)
            [
                *REINIT_OFF,
                "-ss",
                f"{cstart:.3f}",
                "-threads",
                str(threads),
                "-i",
                str(src),
                *out,
                cpu_vf,
            ],
            duration - cstart,
        ),
    }
    stop = threading.Event()  # one half failed or the run is stopped: the other stops too
    found: dict[str, list[float]] = {}
    errors: dict[str, BaseException] = {}

    def half(name: str) -> None:
        proxy, args, seconds = halves[name]

        def command(download: bool) -> list[str]:  # only the GPU half runs again (no hwdownload)
            given = gpu_args(True) if download else args
            return ["ffmpeg", "-y", "-v", "error", *given, *enc, str(proxy)]

        try:
            _run_scene_proxy(command, proxy, seconds, f"scene proxy {src.name} ({name} part)", stop)
            found[name] = _scan(proxy, stop)
        except BaseException as exc:  # noqa: BLE001 - reported by the caller's thread
            errors[name] = exc
            stop.set()
        finally:
            with contextlib.suppress(OSError):
                proxy.unlink(missing_ok=True)

    logger.info(
        "%s: scene decode split at %s: NVDEC before, the CPU (%d threads) after",
        src.name,
        clock(split),
        threads,
    )
    workers = [
        threading.Thread(target=half, args=(n,), name=f"scene-{n}", daemon=True) for n in halves
    ]
    try:
        for w in workers:
            w.start()
        while any(w.is_alive() for w in workers):
            for w in workers:
                w.join(SCENE_POLL_S)  # in steps: Ctrl+C gets through on Windows
            if cancel is not None and cancel.is_set():
                stop.set()
    except BaseException:
        stop.set()
        for w in workers:
            w.join(SCENE_JOIN_S)
        raise
    finally:
        for proxy, _args, _s in halves.values():  # a half still busy after the join
            with contextlib.suppress(OSError):
                proxy.unlink(missing_ok=True)
    if cancel is not None and cancel.is_set():
        raise RunCancelled(f"scene detection {src.name}")
    if errors:
        if cancel is not None and cancel.wait(STOP_GRACE_S):  # Ctrl+C reached ffmpeg first
            raise RunCancelled(f"scene detection {src.name}") from None
        if cancel is None:
            time.sleep(STOP_GRACE_S)  # a Ctrl+C that reached ffmpeg first is raised here
        # the half that failed first; the other one was stopped (RunCancelled) because of it
        real = [e for e in errors.values() if not isinstance(e, RunCancelled)]
        first = (real or list(errors.values()))[0]
        reason = str(first).strip().splitlines()[-1:] or [type(first).__name__]
        logger.warning(
            "%s: the split scene decode failed (%s); decoding it in one pass",
            src.name,
            reason[0],
        )
        return None
    return _merge_seam(found["gpu"], found["cpu"], split, cstart)


def _make_proxy(
    src: Path,
    proxy: Path,
    args: tuple[list[str], list[str], str, list[str]],
    seconds: float,
    cancel: threading.Event | None = None,
    gpu: bool = True,
) -> ProgressResult:
    """Encode the scene proxy ``proxy`` of ``src`` with the :func:`_proxy_args` ``args``
    (``seconds`` long, for the progress lines): on the GPU decoder when there is one, else or
    when that fails on the CPU; ``gpu`` False always on the CPU. A ``cancel`` (see
    :func:`scene_cuts`) or a failure removes the partial proxy. Returns the last ffmpeg run's
    result (its stderr may name a read error ffmpeg did not fail on)."""
    pre, post, vf, enc = args
    label = f"scene proxy {src.name}"

    def command(dec: HwDecoder | None, download: bool = False) -> list[str]:
        dec_args, chain = _scene_decode(dec, vf, src, download)  # input options go before -i
        args = [*pre, *dec_args, "-i", str(src), *post]
        return ["ffmpeg", "-y", "-v", "error", *args, "-an", "-vf", chain, *enc, str(proxy)]

    # NVDEC or VAAPI when probed here; a failed GPU run decodes on the CPU
    hw = decoder_for(src) if gpu else None
    opts = {"cancel": cancel} if cancel is not None else {}  # plain call when not overlapped
    try:
        try:
            res = _run_scene_proxy(
                lambda download: command(hw, download), proxy, seconds, label, cancel
            )
        except FFmpegError as err:
            if cancel is not None and cancel.wait(STOP_GRACE_S):  # Ctrl+C reached ffmpeg too
                raise RunCancelled(label) from None
            if hw is None:
                raise
            # message: ffmpeg's header line, then the tail of its stderr; show its first line
            lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
            first = (lines[1] if len(lines) > 1 else lines[0] if lines else "")[:200]
            logger.debug(
                "scene proxy: GPU decode failed for %s (%s); re-running on the CPU", src.name, first
            )
            proxy.unlink(missing_ok=True)  # the re-run starts clean
            res = run_progress(command(None), seconds, label, **opts)
        if cancel is not None and cancel.is_set():
            raise RunCancelled(label)
    except BaseException:  # a partial lossless proxy is large: never leave it behind
        proxy.unlink(missing_ok=True)
        raise
    return res


class _SceneJob:
    """:func:`scene_cuts` of one source in a thread, alongside its Whisper run (#725).

    ``cancel`` stops it: ffmpeg is killed (``run_progress``) or the detection stopped, and no
    :data:`SCENES_FILE` is written.
    An exception of the detection is kept and raised by :meth:`result`, in the caller's thread."""

    def __init__(
        self,
        src: Path,
        cdir: Path,
        duration: float,
        offset: float,
        limit: float | None,
        cpu_free: bool = False,
        readable_from: float | None = None,
    ) -> None:
        self.cancel = threading.Event()
        self.cuts: list[float] = []
        self.took = 0.0  # seconds the thread ran
        self._error: BaseException | None = None
        self._args = (src, cdir, duration, offset, limit)
        # the split decode (scene_cuts' cpu_free): a plain call when Whisper is on the CPU
        self._opts: dict[str, Any] = {"cpu_free": True} if cpu_free else {}
        if readable_from is not None:  # a recovered part's lead-in (scene_cuts)
            self._opts["readable_from"] = readable_from
        self._thread = threading.Thread(target=self._main, name="scene-overlap", daemon=True)
        self._thread.start()

    def _main(self) -> None:
        t0 = time.perf_counter()
        try:
            self.cuts = scene_cuts(*self._args, cancel=self.cancel, **self._opts)
        except BaseException as exc:  # noqa: BLE001 - re-raised by result()
            self._error = exc
        self.took = round(time.perf_counter() - t0, 2)

    def result(self) -> list[float]:
        """Wait for the cuts (or the thread's exception, raised here)."""
        while self._thread.is_alive():
            self._thread.join(SCENE_POLL_S)
        if self._error is not None:
            raise self._error
        return self.cuts

    def stop(self) -> None:
        """Cancel and wait for the thread, for a bounded time. Idempotent."""
        self.cancel.set()
        self._thread.join(SCENE_JOIN_S)
        if self._thread.is_alive():
            logger.debug("scene thread still busy after %.0fs; left to exit", SCENE_JOIN_S)


def scores_scenes(style: Style) -> bool:
    """True when ``style`` weighs the scene curve. A style that does not (``funny``,
    ``commentary``, a user style whose weights give ``scene`` 0) uses scene cuts only to snap
    candidate edges, so they are detected only around those edges (:func:`edge_windows`)."""
    return bool(style.weights.get("scene"))


def edge_windows(style: Style, span: float) -> bool:
    """True when the scene cuts of a part ``span`` seconds long are detected only around its
    candidate edges (:class:`EdgeCuts`, #730): ``style`` does not weigh them
    (:func:`scores_scenes`) and the part is not longer than :data:`LONG_FILE_S`. A longer part's
    full pass decodes keyframes only (:func:`_proxy_args`), faster than the windows' 10 fps
    decode, so it keeps the full pass (:data:`SCENES_FILE`)."""
    return not scores_scenes(style) and span <= LONG_FILE_S


def _edge_window(t: float, span: float) -> tuple[float, float]:
    """The decode window of the candidate edge ``t`` of a part ``span`` long:
    :data:`WINDOW_PAD_S` either side, its start on the full proxy's frame grid
    (:data:`SCENE_GRID_S`) so the window's frames are those of a full pass."""
    a = math.floor(max(0.0, t - WINDOW_PAD_S) / SCENE_GRID_S + 1e-6) * SCENE_GRID_S
    return round(a, 3), round(min(span, t + WINDOW_PAD_S), 3)


def _window_cuts(io: Path, cdir: Path, offset: float, a: float, b: float) -> list[float]:
    """Scene cuts of ``[a, b]`` of the part at ``offset`` of ``io`` (part-relative time), from a
    proxy of that stretch made as the full one of a normal-length file (:func:`_proxy_args`),
    always decoded on the CPU: each window is a short ffmpeg run, and a GPU decoder's start-up
    costs more than the few seconds it decodes. The seek counts from the container start, as the
    full pass's time does (#976), so ``io`` must seek exactly (:func:`_seeks_exactly`)."""
    # its own name: another run (project) may read the same source from the shared cache folder
    proxy = cdir / f"proxy-window-{os.getpid()}-{threading.get_ident()}.mp4"
    try:
        _make_proxy(io, proxy, _proxy_args(b - a, offset + a, b - a), b - a, gpu=False)
        found = _scan(proxy, None)  # its video handle is closed on return
    finally:
        try:
            proxy.unlink(missing_ok=True)
        except OSError as exc:  # never replace the original exception
            logger.debug("could not delete %s: %s", proxy, exc)
    return [round(a + t, 3) for t in found]


class EdgeCuts:
    """The scene cuts of one part (or recovered part) of a style that does not weigh the scene
    curve, up to :data:`LONG_FILE_S` long (:func:`edge_windows`), detected only in windows
    around candidate edges (#730), decoded on the CPU (:func:`_window_cuts`).

    :func:`~cutpoint.analyze._snap` moves an edge to a cut at most :data:`SNAP_REACH_S` away, and
    the detector needs :data:`SCENE_WARMUP_S` of picture around a cut, so a window reaches
    :data:`WINDOW_PAD_S` either side of an edge. The windows decoded and their cuts are kept in the
    part's cache folder as :data:`SCENE_WINDOWS_FILE` (part-relative time, as :data:`SCENES_FILE`):
    a later run decodes only the windows not covered yet and adds them. The saved windows never
    overlap: a new window that touches saved ones is decoded together with them, so every cut comes
    from one decode that saw the whole window. A full :data:`SCENES_FILE` of the part (a full pass,
    for a style that weighs scenes) is always used instead, and a container that does not seek
    exactly (:func:`_seeks_exactly`) gets the full pass instead of windows."""

    def __init__(
        self,
        io: Path,
        cdir: Path,
        span: float,
        offset: float = 0.0,
        limit: float | None = None,
        on_time: Callable[[float], None] | None = None,
    ) -> None:
        self.io, self.cdir, self.span, self.offset, self.limit = io, cdir, span, offset, limit
        self.on_time = on_time  # gets the seconds each decode took (``scenes_s``)

    def _saved(self) -> tuple[list[tuple[float, float]], list[float]]:
        data = cache.load_json(self.cdir / SCENE_WINDOWS_FILE)
        try:
            if data["version"] != SCENE_WINDOWS_VERSION:  # another clock (#976)
                return [], []
            wins = [(float(a), float(b)) for a, b in data["windows"]]
            return wins, [float(x) for x in data["cuts"]]
        except (TypeError, KeyError, ValueError):  # none yet, or not this file's shape
            return [], []

    def around(self, edges: Iterable[float]) -> list[float]:
        """All the cuts known, once the windows of ``edges`` (part-relative) are decoded."""
        full = cache.load_json(self.cdir / SCENES_FILE)
        if full is not None:
            return list(full)
        wins, cuts = self._saved()
        eps = 1e-6
        need = [
            w
            for w in (_edge_window(t, self.span) for t in edges)
            if w[1] > w[0] and not any(a <= w[0] + eps and w[1] - eps <= b for a, b in wins)
        ]
        if not need:
            return sorted(cuts)
        if not _seeks_exactly(self.io):  # MPEG-TS and the like: a window would seek inexactly
            logger.info(
                "%s: no exact seek in this container; detecting scene cuts in one pass",
                self.io.name,
            )
            return self.full()
        t0 = time.perf_counter()
        todo = intervals.merge(need)
        logger.info(
            "%s: detecting scene cuts around the candidate edges (%d windows, %.0fs)",
            self.io.name,
            len(todo),
            sum(b - a for a, b in todo),
        )
        for a, b in todo:
            touch = [w for w in wins if w[0] <= b and a <= w[1]]
            a, b = min([a, *(w[0] for w in touch)]), max([b, *(w[1] for w in touch)])
            found = _window_cuts(self.io, self.cdir, self.offset, a, b)
            wins = sorted([*(w for w in wins if w not in touch), (a, b)])
            cuts = sorted([*(x for x in cuts if not a <= x <= b), *found])
            cache.save_json(
                self.cdir / SCENE_WINDOWS_FILE,
                {
                    "version": SCENE_WINDOWS_VERSION,
                    "windows": [list(w) for w in wins],
                    "cuts": cuts,
                },
            )
        if self.on_time is not None:
            self.on_time(time.perf_counter() - t0)
        return cuts

    def full(self) -> list[float]:
        """The full pass (:func:`scene_cuts`, cached as :data:`SCENES_FILE`): for a style that
        weighs the scene curve after all (``--style auto`` decided after the transcript)."""
        t0 = time.perf_counter()
        cuts = scene_cuts(self.io, self.cdir, self.span, self.offset, self.limit)
        if self.on_time is not None:
            self.on_time(time.perf_counter() - t0)
        return cuts
