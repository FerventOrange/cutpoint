"""``cutpoint analyze``: find and rank highlight candidates in one or more videos.

Signals per source (all on a 0.1 s grid, each normalised to 0-1):
    loudness  - robust z-score of ebur128 momentary loudness vs the clip's baseline
    speech    - style keywords/exclamations from the faster-whisper transcript
    laughter  - laughter tokens in the transcript
    scene     - density of PySceneDetect cuts (from a small, low-fps proxy); a style that does
                not weigh it detects cuts only around candidate edges (parts up to
                LONG_FILE_S), to snap them (#730)
    continuity- fraction of the surrounding ~10 s covered by speech (commentary)
Each is a feature of :mod:`cutpoint.features`; their weighted sum (:mod:`cutpoint.scorer`) is
peak-picked into windows, merged, snapped to scene cuts and word boundaries, scored 0-100 over
the available features, optionally re-scored by a vision model, and written to
``highlights.json``.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import importlib.util
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from cutpoint import (
    cache,
    category,
    config,
    default_fit,
    features,
    folders,
    jsonout,
    paths,
    progress,
    runstate,
    scorer,
)
from cutpoint.config import path_key, remap_source, unc_path
from cutpoint.dsp import conv_same
from cutpoint.errors import FFmpegError, LowMemoryError
from cutpoint.events import paths as event_paths
from cutpoint.events.menu_spans import drop_menu_time
from cutpoint.events.scoring import (
    Seed,
    heard_laughter,
    is_nominated,
    number_candidates,
    recorded_style,
    rescore,
    restore_widened,
)
from cutpoint.events.setup import COMPONENTS, EVENTS_MODULES
from cutpoint.ffmpeg_utils import (
    NVDEC_DOWNLOAD,
    READ_ERROR_RE,
    STOP_GRACE_S,
    HwDecoder,
    MediaInfo,
    ProgressResult,
    RunCancelled,
    decoder_for,
    extract_audio,
    ffprobe_json,
    first_frame_time,
    gpu_chain,
    momentary_loudness,
    nvdec_download,
    probe,
    read_window_ok,
    run_progress,
    single_video_stream,
    source_tracks,
    wav_duration,
)
from cutpoint.progress import ANALYZE_STEPS, LABELS, SKIPPABLE
from cutpoint.sources import READ_ERRORS_FILE, load_read_errors, source_segments
from cutpoint.styles import (
    ACTION_CAM,
    AUTO,
    LAUGHTER,
    PROFANITY,
    AutoRequest,
    Style,
    style_hash,
)
from cutpoint.transcribe import (
    Segment,
    Transcriber,
    cached_transcript,
    default_model,
    filter_hallucinations,
    flat_markers,
    flat_words,
    is_fallback,
    record_transcript,
    split_markers,
    transcript_name,
)
from cutpoint.utils import clock, find_videos, matches_glob, slugify
from cutpoint.vision import PRIOR_N, RATING_PRIOR, VisionRater, speech_excerpt

if TYPE_CHECKING:
    from cutpoint.activity import Prompts
    from cutpoint.events.ocr_prefetch import OcrPrefetch

logger = logging.getLogger(__name__)

DT = 0.1  # analysis grid step (s), matches ebur128's 10 Hz momentary output
LONG_FILE_S = 1800  # above this, scene detection decodes keyframes only
PROFANE = {p.rstrip("*") for p in PROFANITY}  # names as keyword_hits reports them
PROFANITY_CUE_S = 3.0  # a profane word counts only if another cue is this close
PROFANITY_CUE_MIN = 0.8  # minimum weight of a keyword that counts as a cue
# the vision band (#909): the default edit plus k below its cut, k = this share of the edit's
# moments, at least VISION_BAND_MIN and at most VISION_BAND_MAX
VISION_BAND_SHARE, VISION_BAND_MIN, VISION_BAND_MAX = 0.3, 2, 24
VISION_SPAN_EPS = 1e-3  # seconds: a candidate span this close to its rated one still matches
PICK_FLOOR = 0.25  # a peak below this share of the strongest one is not a candidate ...
LONG_LIST_FLOOR = 0.5  # ... and, past the usual count (:func:`usual_count`), not below this share
SNIPPET_CHARS = 240  # a candidate's transcript snippet
UNSURE_P = 0.5  # transcript words Whisper heard with less confidence are listed as unsure
SHORTFALL_S, SHORTFALL_FRAC, SHORTFALL_FLOOR_S = 2.0, 0.01, 0.5  # truncation thresholds
RESUME_STEPS_S = (5, 10, 20, 40, 80, 160, 320)  # probe offsets past a read failure
EXTRACT_SHARE = 0.1  # the audio extract's share of a source in the transcribe step's pct
EXTRACT_POLL_S = 0.5  # how often the growing WAV is measured for that pct
WAV_BYTES_PER_S = 16000 * 2  # extract_audio's 16 kHz mono s16 output
WAV_HEADER = 44
# The audio tracks the analysis WAV mixes, saved next to it only when copies of another track
# are left out (#685); without it the WAV mixes every track
MIX_FILE = "audio16k-tracks.json"
# The caches made from the analysis WAV and, last, the WAV, removed when it mixed other tracks
AUDIO_CACHES = ("loudness.json", "transcript-*.json", "fallback-transcript-*.json",
                "events-audio-*.json", "voice-*.npz", "audio16k.wav")  # fmt: skip
RESUME_REFINE = 2  # bisection probes to lose less footage before the resume point
RESUME_MARGIN_S = 1.0  # start the recovered part a little after the first good probe
MAX_RECOVERED_PARTS = 5  # per source
# Result files of the project written only when analyze completes, all in one commit
# (cache.hold_outputs): a failed or cancelled run leaves the previous ones as they were and a
# first one none (#408). The steps read them from memory meanwhile. Kept as written on
# purpose: discovery/ (HUD discovery's proposal, for the preset it installs in the user's
# preset folder at once) and the caches outside the project.
HELD_OUTPUTS = (
    "transcript.txt",
    "highlights.json",
    "highlights.pre-events.json",  # deleted once ranked with the events
    "events.json",
    "events.txt",
    "crashes.json",
    "crash_check",  # the crash contact sheets crashes.json names
    "activity.json",
    "telemetry.json",  # action-cam
    "action_cam_features",
)


@dataclass
class Candidate:
    """A highlight window inside one source."""

    source: Path
    start: float
    end: float
    peak_t: float
    peak: float
    signals: dict[str, float]
    keywords: list[str] = field(default_factory=list)
    label_hint: str = ""
    transcript: str = ""
    transcript_unsure: list[tuple[int, float]] | None = None  # see _transcript_snippet
    score: float = 0.0
    label: str = ""
    kind: str = ""  # action-cam: dominant feature (airtime, crash, ...); a crash moment: crash
    must_include: bool = False  # a confident crash (cutpoint clips always exports it)
    event: dict[str, Any] | None = None  # the crash behind a crash moment (or one it overlaps)
    nominated: list[str] = field(default_factory=list)  # event types that seeded it (nominate)
    whole: bool = False  # a dialogue stretch: its span stays as marked (no max_len, no merging)
    over: bool = (
        False  # a long dialogue stretch: stays whole over the existing candidates (nominate)
    )


# --------------------------------------------------------------------------- signals
def _gauss_kernel(sigma_s: float) -> tuple[int, np.ndarray]:
    """Unnormalised Gaussian kernel over +-3 sigma at the analysis rate: ``(radius, k)``."""
    radius = int(3 * sigma_s / DT)
    return radius, np.exp(-0.5 * (np.arange(-radius, radius + 1) * DT / sigma_s) ** 2)


def _gauss_smooth(x: np.ndarray, sigma_s: float) -> np.ndarray:
    _, k = _gauss_kernel(sigma_s)
    return conv_same(x, k / k.sum())


def _impulse_signal(n: int, events: list[tuple[float, float]], sigma_s: float = 1.0) -> np.ndarray:
    """Place weighted impulses, blur them (peak height == weight), then saturate to 0-1."""
    x = np.zeros(n)
    for t, w in events:
        i = int(t / DT)
        if 0 <= i < n:
            x[i] += w
    _, k = _gauss_kernel(sigma_s)
    return 1.0 - np.exp(-conv_same(x, k))


def loudness_signal(lufs: np.ndarray) -> np.ndarray:
    """Positive robust z-score of loudness, mapped to 0-1 (z=4 -> ~1)."""
    x = np.maximum(lufs, -70.0)  # silence floor
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    z = np.clip((x - med) / max(mad, 1.0), 0, None)
    return np.clip(_gauss_smooth(z, 0.4) / 4.0, 0, 1)


def continuity_signal(n: int, words: list[dict[str, Any]], window_s: float = 10.0) -> np.ndarray:
    """Fraction of a sliding window covered by spoken words (0-1)."""
    talk = np.zeros(n)
    for w in words:
        talk[int(w["start"] / DT) : int(w["end"] / DT) + 1] = 1.0
    k = int(window_s / DT)
    return np.clip(conv_same(talk, np.ones(k) / k) * 1.25, 0, 1)


def _norm_token(word: str) -> str:
    return re.sub(r"[^a-z0-9*]", "", word.lower())


def keyword_hits(
    words: list[dict[str, Any]], patterns: dict[str, float], *, once: bool = False
) -> list[tuple[float, float, str]]:
    """Find (time, weight, phrase) matches; ``*`` suffix = prefix match. ``once``: a word that
    several patterns match (``haha`` for ``haha*`` and ``hah*``) counts once, for the strongest."""
    compiled: dict[tuple[str, ...], tuple[float, str]] = {}
    for phrase, weight in patterns.items():
        parts = tuple(_norm_token(p) for p in phrase.split())
        compiled[parts] = (weight, phrase.rstrip("*"))
    toks = [_norm_token(w["word"]) for w in words]
    hits = []
    for i in range(len(toks)):
        found = []
        for parts, (weight, phrase) in compiled.items():
            if i + len(parts) > len(toks):
                continue
            if all(
                toks[i + j].startswith(p[:-1]) if p.endswith("*") else toks[i + j] == p
                for j, p in enumerate(parts)
            ):
                found.append((float(words[i]["start"]), weight, phrase))
        hits += [max(found, key=lambda h: h[1])] if once and found else found
    return hits


# Whisper's non-speech laughter markers: "[laughter]", "(laughs)", "*laughing*", "[light laughter]".
# The brackets (or asterisks) are required: the plain spoken word is not laughter.
_LAUGH_STEM = r"(?:laugh|giggl|chuckl|chortl|snicker|guffaw)"
_LAUGH_MARKER = re.compile(
    rf"\[[^\[\]()*]{{0,30}}?\b{_LAUGH_STEM}\w*[^\[\]()*]{{0,30}}\]"
    rf"|\([^\[\]()*]{{0,30}}?\b{_LAUGH_STEM}\w*[^\[\]()*]{{0,30}}\)"
    rf"|\*{_LAUGH_STEM}\w*\*",
    re.IGNORECASE,
)


def laughter_hits(
    words: list[dict[str, Any]], markers: list[dict[str, Any]] | None = None
) -> list[tuple[float, float, str]]:
    """Find the (time, weight, phrase) laughter hits of the transcript ``words``: the
    interjections of :data:`~cutpoint.styles.LAUGHTER` (haha, lol) and Whisper's non-speech
    markers for laughing (``[laughter]``, ``(laughs)``, ``*laughing*``; weight 1, the phrase is
    the text inside the brackets), found among the ``words`` and in ``markers``
    (:func:`~cutpoint.transcribe.flat_markers`: the markers taken out of the words). The spoken
    word "laughter" or "laugh" is not a hit."""
    hits = keyword_hits(words, LAUGHTER, once=True)
    text, at = "", []  # the words joined by spaces, and the word each character belongs to
    for i, w in enumerate(words):
        piece = str(w["word"]).strip()
        at += [i] * (len(piece) + 1)
        text += piece + " "
    for m in _LAUGH_MARKER.finditer(text):
        phrase = m.group(0)[1:-1].strip().lower()
        hits.append((float(words[at[m.start()]]["start"]), 1.0, phrase))
    for mark in markers or []:
        if m := _LAUGH_MARKER.search(str(mark["text"])):
            hits.append((float(mark["start"]), 1.0, m.group(0)[1:-1].strip().lower()))
    return sorted(hits, key=lambda h: h[0])


def keyword_signal(
    n: int, hits: list[tuple[float, float, str]], cap: float | None, sigma_s: float = 1.0
) -> np.ndarray:
    """Keyword excitement curve (0-1).

    Without ``cap`` this is ``_impulse_signal`` (all nearby hits add up). With a
    cap, at each moment the strongest blurred hit counts fully, the others at
    30%, and the total is clipped to ``cap`` before saturating, so a stream of
    filler words ("nice", "bro", "dude") cannot outrank one "let's go".
    """
    if cap is None:
        return _impulse_signal(n, [(t, w) for t, w, _ in hits], sigma_s)
    radius, k = _gauss_kernel(sigma_s)
    total, strongest = np.zeros(n), np.zeros(n)
    for t, w, _ in hits:
        i = int(t / DT)
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        if lo >= hi:
            continue
        contrib = w * k[lo - (i - radius) : hi - (i - radius)]
        total[lo:hi] += contrib
        np.maximum(strongest[lo:hi], contrib, out=strongest[lo:hi])
    value = np.minimum(cap, strongest + 0.3 * (total - strongest))
    return 1.0 - np.exp(-value)


def gate_profanity(
    kw_hits: list[tuple[float, float, str]],
    laugh_hits: list[tuple[float, float, str]],
    loudness: np.ndarray,
) -> list[tuple[float, float, str]]:
    """Drop profane hits with no other cue (strong keyword, laughter, loudness spike)."""
    cues = [t for t, w, p in kw_hits if p not in PROFANE and w >= PROFANITY_CUE_MIN]
    cues += [t for t, _, _ in laugh_hits]
    cue_arr = np.asarray(cues, dtype=float)
    kept = []
    for t, w, p in kw_hits:
        if p in PROFANE:
            i = min(len(loudness) - 1, max(0, int(t / DT)))
            near_cue = cue_arr.size > 0 and float(np.min(np.abs(cue_arr - t))) <= PROFANITY_CUE_S
            if not near_cue and loudness[i] < 0.5:
                continue
        kept.append((t, w, p))
    return kept


# --------------------------------------------------------------------------- per-source caching
def _loudness(wav: Path, cdir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = cdir / "loudness.json"
    data = cache.load_json(path)
    if data is None:
        times, values = momentary_loudness(wav)
        data = {"t": times, "m": values}
        cache.save_json(path, data)
    # ebur128 reports at the window end; shift to centre of its 400 ms window.
    return np.asarray(data["t"]) - 0.2, np.asarray(data["m"])


def _transcript(
    wav: Path,
    cdir: Path,
    tr: Transcriber,
    on_progress: Callable[[float], None] | None = None,
    src: Path | None = None,
) -> list[Segment]:
    """The transcript of the WAV ``wav`` (cached in ``cdir``); a CPU fallback's, made or reused,
    adds ``src`` (default ``cdir``) to ``tr.fallbacks`` (#892)."""
    path = cached_transcript(tr, cdir)  # without --model also a marked CPU fallback's
    data = cache.load_json(path) if path is not None else None
    if data is not None and path is not None and is_fallback(tr, path):
        logger.info(
            "Reusing %s, made by a CPU fallback of an earlier run without --model "
            "(--model %s transcribes it again)",
            path.name,
            tr.default_name,
        )
    if data is None:
        data = tr.transcribe(wav, on_progress) if on_progress else tr.transcribe(wav)
        # the model that ran: without --model, a CPU fallback runs the CPU default (small)
        path = cdir / transcript_name(tr.model_name, tr.language)
        cache.save_json(path, data)
        record_transcript(tr, path)
    if is_fallback(tr, path):
        tr.fallbacks.append(src or cdir)
    return data


def _warn_transcript_fallback(tr: Transcriber) -> None:
    """The WARNING and ``--json`` ``transcript_fallback`` warning naming the sources whose
    transcript this run is a CPU fallback's (#892)."""
    found = list(dict.fromkeys(getattr(tr, "fallbacks", ())))
    if not found:
        return
    message = (
        f"{len(found)} source(s) transcribed with Whisper {default_model('cpu')} on the CPU "
        f"instead of Whisper {tr.default_name}, as the GPU was short of memory or failed: "
        f"{', '.join(p.name for p in found)}. The speech-based moments can be weaker; after a "
        f"memory shortage, a run with Whisper {tr.default_name} fitting on the GPU (close the "
        "programs using it) transcribes them again"
    )
    logger.warning("%s", message)
    jsonout.warn("transcript_fallback", message, found)


def _mixed_tracks(cdir: Path, streams: int) -> list[int] | None:
    """The tracks :data:`MIX_FILE` in ``cdir`` says its WAV mixes; None without a valid one
    (the WAV mixes every track)."""
    mix = cache.load_json(cdir / MIX_FILE)
    have = mix.get("tracks") if isinstance(mix, dict) else None
    if not isinstance(have, list) or not have:
        return None
    return have if all(isinstance(i, int) and 0 <= i < streams for i in have) else None


def _analysis_tracks(src: Path, io: Path, info: MediaInfo, cdir: Path) -> tuple[int, ...]:
    """Audio tracks the analysis WAV of ``src`` (read as ``io``) mixes, with ``cdir`` made to
    hold only audio caches of that mix (:func:`_audio_mix`). When the tracks could not be
    compared, the cached WAV's tracks (or every track) are used and nothing is removed."""
    found = source_tracks(io, info)  # copies of another track left out (#685)
    if found is None:
        have = _mixed_tracks(cdir, info.audio_streams)
        return tuple(range(info.audio_streams)) if have is None else tuple(have)
    _audio_mix(src, cdir, found, info.audio_streams)
    return found


def _audio_mix(src: Path, cdir: Path, tracks: tuple[int, ...], streams: int) -> None:
    """Make ``cdir`` hold only audio caches of a WAV mixing ``tracks`` (of ``streams``).

    A WAV made before duplicate tracks were left out (#685), or with other tracks, is
    removed with every cache made from it (:data:`AUDIO_CACHES`), so it is extracted and
    analysed again once. :data:`MIX_FILE` records ``tracks`` when they are not all
    ``streams``; a WAV mixing every track (one track included) has none, so its folder
    stays as it was. A file that cannot be removed (open in another program) is named in a
    WARNING and the record is left as it was, so the next run tries again; the WAV is then
    kept too (removed last, only once every cache made from it is gone), so the old WAV and
    the caches left are used together meanwhile.
    """
    want = list(tracks) if len(tracks) < streams else None
    if _mixed_tracks(cdir, streams) == want:
        return
    stale = [f for pattern in AUDIO_CACHES for f in cdir.glob(pattern)]
    kept: list[Path] = []
    for f in stale:
        if kept and f.name == "audio16k.wav":
            continue  # an unfinished purge keeps the old WAV its remaining caches match
        try:
            f.unlink(missing_ok=True)
        except OSError as err:
            kept.append(f)
            logger.debug("removing %s failed: %s", f, err)
    if kept:
        logger.warning(
            "%s: could not remove %s (open in another program?); its analysis uses the old "
            "audio this time and is made again on the next run",
            src.name,
            ", ".join(f.name for f in kept),
        )
        return
    if stale:  # the WAV (removed last) is gone: extracted again
        logger.info(
            "%s: the cached analysis audio mixed other tracks; extracting it again", src.name
        )
    if want is None:
        (cdir / MIX_FILE).unlink(missing_ok=True)
    else:
        cache.save_json(cdir / MIX_FILE, {"tracks": want})


SOURCE_STAGES = ("probe_s", "audio_extract_s", "loudness_s", "transcribe_s", "scenes_s")
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


def scene_overlap() -> bool:
    """True when ``CUTPOINT_ANALYZE_SCENE_OVERLAP=1``: a source's scene cuts are detected
    while Whisper transcribes it."""
    return (paths.env(SCENE_OVERLAP_ENV) or "").strip() == "1"


def _source_timings(results: Sequence[SourceResult], prefetch: OcrPrefetch | None) -> list[Any]:
    """``timings.sources`` (#725): the seconds each source (or recovered part) spent in each
    stage; ``ocr_wait_s`` is the time the main thread waited for the background OCR read
    (0 without one), counted on the first part of a source."""
    waits = dict(prefetch.waits) if prefetch is not None else {}
    out: list[Any] = []
    for r in results:
        rec: dict[str, Any] = {"source": str(r.source)}
        if r.offset:
            rec["offset"] = r.offset
        for name in SOURCE_STAGES:
            rec[name] = r.stages.get(name, 0.0)
        rec["ocr_wait_s"] = round(waits.pop(str(r.source), 0.0), 2)
        rec["cached"] = bool(r.stages.get("cached", False))
        if r.stages.get("overlap"):  # the scene cuts ran alongside the transcription
            rec["overlap"] = True
        if r.stages.get("scene_windows"):  # scenes_s: windows around the candidate edges only
            rec["scene_windows"] = True
        out.append(rec)
    return out


def _cached_transcripts(sources: Sequence[Path], tr: Transcriber | None) -> int:
    """How many of ``sources`` have their (whole-file) transcript of ``tr`` cached;
    unreachable ones count as not cached, and none without a transcriber."""
    if tr is None:
        return 0
    n = 0
    for src in sources:
        with contextlib.suppress(OSError):
            cdir = cache.source_dir(src)
            n += cached_transcript(tr, cdir) is not None
    return n


@contextlib.contextmanager
def _watch_extract(
    wav: Path, seconds: float, report: Callable[[float], None] | None
) -> Iterator[None]:
    """While the block runs :func:`extract_audio` into ``wav``, report the share of its
    ``seconds`` written so far (0-1), measured from the size of the growing temp WAV."""
    if report is None or seconds <= 0:
        yield
        return
    tmp = wav.with_suffix(".tmp.wav")  # extract_audio's temp file, renamed when complete
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(EXTRACT_POLL_S):
            try:
                size = tmp.stat().st_size
            except OSError:  # not created yet
                continue
            report(min(1.0, max(0, size - WAV_HEADER) / (WAV_BYTES_PER_S * seconds)))

    thread = threading.Thread(target=poll, name="extract-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def _first_shift(src: Path, offset: float, duration: float) -> float | None:
    """Where on the container clock (from :attr:`MediaInfo.fstart`) the first frame a whole
    scene pass of ``src`` decodes lies (:func:`_proxy_args`), or None when ffmpeg's own count
    is that clock: a pass that seeks (only in a container that seeks exactly), a whole pass of
    such a container, and a source ffprobe or the one-frame decode fails on (the pass then
    counts as ffmpeg does)."""
    if offset > 0 or _seeks_exactly(src):
        return None
    try:
        first = first_frame_time(src, keyframes=duration > LONG_FILE_S)
        return first - probe(src).fstart
    except FFmpegError as exc:
        logger.debug("%s: no first frame time (%s); the scene pass counts as ffmpeg does", src, exc)
        return None


REINIT_OFF = ["-reinit_filter", "0"]  # input option: one filter graph per pass (_proxy_args)


def _proxy_args(
    duration: float,
    offset: float = 0.0,
    limit: float | None = None,
    shift: float | None = None,
) -> tuple[list[str], list[str], str, list[str]]:
    """The scene proxy's input options, output options, filter chain and encoder arguments.

    Every pass counts from the container start (#976), so its cuts are on the clock clips are
    cut on: ``fps`` with ``start_time=0`` makes the proxy's first frame the seek point (the
    container start without one) and pads a video that starts later with its first frame.
    That is ffmpeg's own count in a container that seeks exactly (:func:`_seeks_exactly`). In
    MPEG-TS ffmpeg 6.1 counts a whole pass from the video instead (``-an``), so the first
    decoded frame is put at ``shift`` (:func:`_first_shift`: its container time, also when the
    file starts in the middle of a GOP); ffmpeg's own timestamps stay, so a timestamp reset
    (joined TS files) still runs on. A seek needs a container that seeks exactly.

    ``-reinit_filter 0`` keeps one filter graph for the whole pass: where the video parameters
    change mid-file (a joined file's second half with other color tags) ffmpeg 9 would rebuild
    it, and ``setpts`` re-reads its start and ``fps`` pads from 0 again, so the timestamps
    after the join are wrong or lost."""
    long_file = duration > LONG_FILE_S
    pre = [*REINIT_OFF, *(["-skip_frame", "nokey"] if long_file else [])]
    if offset > 0:
        pre += ["-ss", f"{offset:.3f}"]
    post = ["-t", f"{limit:.3f}"] if limit else []
    rate, width = (2, 256) if long_file else (10, 320)
    vf = f"fps={rate}:start_time=0,scale={width}:-2"
    if shift is not None and offset <= 0:
        vf = f"setpts=PTS-STARTPTS+{shift:.6f}/TB,{vf}"
    # Lossless: a lossy proxy has a keyframe every 250 frames (25 s at 10 fps) that adds to the
    # detector's content value, so cuts appeared there and moved with the pass's start (#994)
    return pre, post, vf, ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]


_SCENE_DOWNLOAD = NVDEC_DOWNLOAD


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


def _scene_decode(hw: HwDecoder | None, vf: str, src: Path) -> tuple[list[str], str]:
    """The decoder's input options and the filter chain of the scene proxy. On a GPU decoder and
    an 8- or 10-bit 4:2:0 source the frames stay on the GPU until the ``fps`` filter
    dropped the ones the detector does not read, so only those are downloaded. Other formats
    (4:4:4, 12-bit, unknown) and the CPU keep the plain path. The proxy's size is fixed
    (:func:`_fixed_scale`).

    A container that does not seek exactly (MPEG-TS, possibly joined files) is decoded without
    a ``hwdownload`` in the chain: NVDEC reinitialises at a join and ffmpeg rebuilds the graph
    despite ``-reinit_filter 0`` (``setpts`` and ``fps`` start again, the encoder drops the
    rest, exit 0). CUDA then downloads the frames itself (``-hwaccel cuda`` alone, the plain
    chain); VAAPI always needs the download, so it decodes on the CPU."""
    vf = _fixed_scale(vf, src)
    if hw is None:
        return [], vf
    if not _seeks_exactly(src):
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
        fmt = _SCENE_DOWNLOAD.get(info.pix_fmt or "")
    if fmt is None:
        return hw.input_args(), gpu_chain(hw, vf) or vf
    head, sep, rest = vf.partition(",scale=")  # the proxy chain's fps filter comes before it
    return hw.input_args(keep_on_gpu=True), f"{head},hwdownload,format={fmt}{sep}{rest}"


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
    shared: Any = None,
    cancel: threading.Event | None = None,
    cpu_free: bool = False,
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
        shared: The :class:`~cutpoint.events.ocr_prefetch.SceneShare` of this source (a
            shared-decode mode): its proxy, waited for while it is made, replaces the decode.
        cancel: For a run in a background thread (:class:`_SceneJob`): once set, ffmpeg is
            killed or the detection stopped, the proxy removed and :class:`~cutpoint.ffmpeg_utils.RunCancelled` raised,
            before any :data:`SCENES_FILE` is written.
        cpu_free: The caller knows no CPU-heavy step (the background HUD OCR, Whisper on the
            CPU) runs meanwhile, so the decode may be split between NVDEC and the CPU
            (:func:`_hybrid_plan`, #730). False (the default): one decode, as before.
    """
    path = cdir / SCENES_FILE
    cached = cache.load_json(path)
    if cached is not None:
        return cached
    logger.info("%s: detecting scene cuts", src.name)
    importlib.import_module("scenedetect")  # not installed: fails before any decode
    dopts = {"cancel": cancel} if cancel is not None else {}  # plain call when not overlapped
    if shared is not None and (made := shared.result()) is not None:
        return _detect(made, path, **dopts)
    plan = _hybrid_plan(src, duration, offset, limit, cpu_free)
    if plan is not None and (cuts := _hybrid_cuts(src, cdir, duration, *plan, cancel)) is not None:
        cache.save_json(path, cuts)
        return cuts
    proxy = cdir / "proxy.mp4"
    gpu = scene_decode_mode() != "cpu"
    args = _proxy_args(duration, offset, limit, _first_shift(src, offset, duration))
    _make_proxy(src, proxy, args, duration, cancel, gpu=gpu)
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
# Containers whose input seek is frame-accurate; MPEG-TS seeks inexactly and moves the CPU half's
# cuts (up to 0.6 s, or loses them)
HYBRID_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".mkv"})
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
    :data:`HYBRID_CODECS` and :data:`HYBRID_PIX_FMTS`, its video starts with the container
    (``vstart`` 0), and :data:`HYBRID_MIN_THREADS` CPU threads are left after the headroom.
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
    return (
        info.vcodec in HYBRID_CODECS
        and info.pix_fmt in HYBRID_PIX_FMTS
        # a late start splits correctly since #976, but the NVDEC half's padded start is not
        # measured on NVDEC yet
        and abs(info.vstart) < 1e-3
    )


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
    halves = {
        "gpu": (
            cdir / f"proxy-gpu-{tag}.mp4",
            [*REINIT_OFF, *dec_args, "-i", str(src), "-t", f"{gend:.3f}", *out, chain],
            gend,
        ),
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
        try:
            cmd = ["ffmpeg", "-y", "-v", "error", *args, *enc, str(proxy)]
            run_progress(cmd, seconds, f"scene proxy {src.name} ({name} part)", cancel=stop)
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
) -> None:
    """Encode the scene proxy ``proxy`` of ``src`` with the :func:`_proxy_args` ``args``
    (``seconds`` long, for the progress lines): on the GPU decoder when there is one, else or
    when that fails on the CPU; ``gpu`` False always on the CPU. A ``cancel`` (see
    :func:`scene_cuts`) or a failure removes the partial proxy."""
    pre, post, vf, enc = args
    label = f"scene proxy {src.name}"

    def command(dec: HwDecoder | None) -> list[str]:  # the decoder's input options go before -i
        dec_args, chain = _scene_decode(dec, vf, src)
        args = [*pre, *dec_args, "-i", str(src), *post]
        return ["ffmpeg", "-y", "-v", "error", *args, "-an", "-vf", chain, *enc, str(proxy)]

    # NVDEC or VAAPI when probed here; a failed GPU run decodes on the CPU
    hw = decoder_for(src) if gpu else None
    opts = {"cancel": cancel} if cancel is not None else {}  # plain call when not overlapped
    try:
        try:
            run_progress(command(hw), seconds, label, **opts)
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
            run_progress(command(None), seconds, label, **opts)
        if cancel is not None and cancel.is_set():
            raise RunCancelled(label)
    except BaseException:  # a partial lossless proxy is large: never leave it behind
        proxy.unlink(missing_ok=True)
        raise


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
        shared: Any,
        cpu_free: bool = False,
    ) -> None:
        self.cancel = threading.Event()
        self.cuts: list[float] = []
        self.took = 0.0  # seconds the thread ran
        self._error: BaseException | None = None
        self._args = (src, cdir, duration, offset, limit, shared)
        # the split decode (scene_cuts' cpu_free): a plain call when Whisper is on the CPU
        self._opts = {"cpu_free": True} if cpu_free else {}
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


def _merge_windows(wins: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """``wins`` sorted, the ones that overlap or touch merged."""
    out: list[tuple[float, float]] = []
    for a, b in sorted(wins):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


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

    :func:`_snap` moves an edge to a cut at most :data:`SNAP_REACH_S` away, and the detector
    needs :data:`SCENE_WARMUP_S` of picture around a cut, so a window reaches
    :data:`WINDOW_PAD_S` either side of an edge. The windows decoded and their cuts are kept in
    the part's cache folder as :data:`SCENE_WINDOWS_FILE` (part-relative time, as
    :data:`SCENES_FILE`): a later run decodes only the windows not covered yet and adds them.
    The saved windows never overlap: a new window that touches saved ones is decoded together with
    them, so every cut comes from one decode that saw the whole window. A full
    :data:`SCENES_FILE` of the part (a full pass, for a style that weighs scenes) is always used
    instead, and a container that does not seek exactly (:func:`_seeks_exactly`) gets the full
    pass instead of windows."""

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
        todo = _merge_windows(need)
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


# --------------------------------------------------------------------------- windowing
def _pick_windows(
    src: Path,
    combined: np.ndarray,
    sig: dict[str, np.ndarray],
    style: Style,
    duration: float,
    max_candidates: int,
    usual: int | None = None,
) -> list[Candidate]:
    """Windows around the highest peaks of ``combined``, strongest first, at most
    ``max_candidates``. A peak must reach :data:`PICK_FLOOR` of the strongest one, and once
    ``usual`` windows are picked (None: never) :data:`LONG_LIST_FLOOR`."""
    score = combined.copy()
    gmax = float(score.max()) if score.size else 0.0
    out: list[Candidate] = []
    while len(out) < max_candidates:
        i = int(score.argmax())
        peak = float(score[i])
        floor = LONG_LIST_FLOOR if usual is not None and len(out) >= usual else PICK_FLOOR
        if peak <= 0.05 or peak < floor * gmax:
            break
        # Region around the peak that stays above 60% of it.
        lo, hi = i, i
        while lo > 0 and combined[lo - 1] >= 0.6 * peak:
            lo -= 1
        while hi < len(combined) - 1 and combined[hi + 1] >= 0.6 * peak:
            hi += 1
        t = i * DT
        start = min(lo * DT, t - style.pre_roll)
        end = max(hi * DT, t + style.post_roll)
        if end - start > style.max_len:  # keep the peak, biased toward lead-up
            lead = style.max_len * style.pre_roll / (style.pre_roll + style.post_roll)
            start, end = t - lead, t - lead + style.max_len
        if end - start < style.min_len:
            end = start + style.min_len
        start, end = max(0.0, start), min(duration, end)
        a, b = int(start / DT), max(int(start / DT) + 1, int(end / DT))
        score[a:b] = 0.0
        score[i] = 0.0  # the peak itself, also when the window ends at it (the last sample)
        out.append(
            Candidate(
                source=src,
                start=start,
                end=end,
                peak_t=t,
                peak=peak,
                signals={k: round(float(v[a:b].max()), 3) for k, v in sig.items()},
            )
        )
    return out


def _absorb_into(prev: Candidate, c: Candidate) -> None:
    """Merge ``c`` into ``prev``: extend its end, keep the higher peak and signals."""
    prev.end = max(prev.end, c.end)
    if c.peak > prev.peak:
        prev.peak, prev.peak_t = c.peak, c.peak_t
    prev.signals = {k: max(v, c.signals.get(k, 0)) for k, v in prev.signals.items()}
    prev.nominated = list(dict.fromkeys([*prev.nominated, *c.nominated]))


def _merge(cands: list[Candidate], style: Style) -> list[Candidate]:
    cands = sorted(cands, key=lambda c: c.start)
    merged: list[Candidate] = []
    for c in cands:
        prev = merged[-1] if merged else None
        if (
            prev
            and not (prev.whole or c.whole)
            and c.start - prev.end <= style.merge_gap
            and max(c.end, prev.end) - prev.start <= style.max_len
        ):
            _absorb_into(prev, c)
            continue
        if prev and c.start < prev.end:  # overlap but too long to merge: trim
            c.start = prev.end
            if c.end - c.start < style.min_len * 0.5:
                continue
        merged.append(c)
    return merged


def sentence_bounds(segments: list[Segment], gap: float = 0.7) -> tuple[list[float], list[float]]:
    """Sentence start/end times: segment edges at terminal punctuation or pauses."""
    starts: list[float] = []
    ends: list[float] = []
    segments = [s for s in segments if s.get("text")]  # not a marker holder
    for i, seg in enumerate(segments):
        prev = segments[i - 1] if i else None
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        words = seg.get("words") or [{"start": seg["start"], "end": seg["end"]}]
        if prev is None or prev["text"].rstrip()[-1:] in ".?!" or seg["start"] - prev["end"] >= gap:
            starts.append(words[0]["start"])
        if nxt is None or seg["text"].rstrip()[-1:] in ".?!" or nxt["start"] - seg["end"] >= gap:
            ends.append(words[-1]["end"])
    return starts, ends


def _in_speech(t: float, segments: list[Segment]) -> bool:
    return any(s["start"] - 0.3 <= t <= s["end"] + 0.3 for s in segments if s.get("text"))


def _gap_before(t: float, words: list[dict[str, Any]]) -> float:
    prev = [w["end"] for w in words if w["end"] <= t]
    return t - max(prev) if prev else 1.0


def _gap_after(t: float, words: list[dict[str, Any]]) -> float:
    nxt = [w["start"] for w in words if w["start"] >= t]
    return min(nxt) - t if nxt else 1.0


def _longest_pause(words: list[dict[str, Any]], lo: float, hi: float) -> tuple[float, float] | None:
    """(end of word before, start of word after) for the longest gap inside [lo, hi]."""
    gaps = [
        (b["start"] - a["end"], a["end"], b["start"])
        for a, b in pairwise(words)
        if lo <= a["end"] <= hi and b["start"] - a["end"] >= 0.25
    ]
    if not gaps:
        return None
    _, end_of_prev, start_of_next = max(gaps)
    return end_of_prev, start_of_next


def _snap_sentences(c: Candidate, segments: list[Segment], duration: float) -> None:
    """Move boundaries that fall inside speech to sentence ends, else the longest pause.

    Padding never exceeds half the silence next to the boundary, so it cannot
    reach into a neighbouring word.
    """
    words = flat_words(segments)
    starts, ends = sentence_bounds(segments)
    if _in_speech(c.start, segments):
        before = [t for t in starts if c.start - 12 <= t <= c.start]
        after = [t for t in starts if c.start < t <= c.start + 6]
        if before or after:
            t = max(before) if before else min(after)
        else:
            pause = _longest_pause(words, c.start - 12, c.start + 6)
            t = pause[1] if pause else c.start
        c.start = max(0.0, t - min(0.2, _gap_before(t, words) / 2))
    if _in_speech(c.end, segments):
        after = [t for t in ends if c.end <= t <= c.end + 15]
        before = [t for t in ends if c.end - 6 <= t < c.end and t > c.start]
        if after or before:
            t = min(after) if after else max(before)
        else:
            pause = _longest_pause(words, c.end - 6, c.end + 15)
            t = pause[0] if pause else c.end
        c.end = min(duration, t + min(0.4, _gap_after(t, words) / 2))


def _snap(
    c: Candidate, cuts: list[float], segments: list[Segment], duration: float, style: Style
) -> None:
    """Snap to nearby scene cuts, then sentence edges (long-form), then word edges."""
    for attr in ("start", "end"):
        t = getattr(c, attr)
        near = [x for x in cuts if abs(x - t) <= SNAP_REACH_S]
        if near:
            setattr(c, attr, min(near, key=lambda x: abs(x - t)))
    if style.snap_sentences and not c.whole:  # a dialogue stretch ends where the reader said
        _snap_sentences(c, segments, duration)
    words = flat_words(segments)
    for w in words:  # words are chronological, so each fix can only move outward
        if w["start"] < c.start < w["end"]:
            c.start = max(0.0, w["start"] - min(0.05, _gap_before(w["start"], words) / 2))
    for w in reversed(words):
        if w["start"] < c.end < w["end"]:
            c.end = min(duration, w["end"] + min(0.15, _gap_after(w["end"], words) / 2))
    c.start, c.end = round(max(0.0, c.start), 3), round(min(duration, c.end), 3)


def _resolve_overlaps(cands: list[Candidate], style: Style) -> list[Candidate]:
    """Snapping can make windows overlap again: merge them, or trim the later one."""
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.start):
        prev = out[-1] if out else None
        if prev is None or c.start >= prev.end:
            out.append(c)
        elif not (prev.whole or c.whole) and (
            max(c.end, prev.end) - prev.start <= style.max_len + min(15.0, 0.25 * style.max_len)
        ):
            _absorb_into(prev, c)
        elif c.end - prev.end >= style.min_len * 0.5:
            c.start = prev.end
            out.append(c)
    return out


def transcript_text(name: str, segments: list[Segment], pause: float = 0.6) -> list[str]:
    """Readable transcript lines, broken at pauses and sentence-ending segment edges.

    Format: ``[name HH:MM:SS.s-HH:MM:SS.s] text``. (No diarization: a speaker
    turn is only detected when it coincides with a pause or segment end.)
    """
    lines: list[str] = []
    cur: list[dict[str, Any]] = []

    def flush() -> None:
        if cur:
            text = " ".join(w["word"] for w in cur)
            lines.append(f"[{name} {clock(cur[0]['start'])}-{clock(cur[-1]['end'])}] {text}")
            cur.clear()

    for seg in segments:
        if not seg.get("text"):  # a marker holder
            continue
        words = seg.get("words") or [
            {"start": seg["start"], "end": seg["end"], "word": seg["text"]}
        ]
        for w in words:
            if cur and (w["start"] - cur[-1]["end"] >= pause or len(cur) >= 60):
                flush()
            cur.append(w)
        if cur and seg["text"].rstrip()[-1:] in ".?!":
            flush()
    flush()
    return lines


def vision_blend(score: float, rating: float | None, weight: float) -> float:
    """``score`` with a 0-10 vision ``rating`` applied as a bounded multiplier
    (``1 + weight * (rating - 5) / 5``, ``weight`` being the style's
    ``vision_weight``), kept within 0-100; an unrated candidate (``rating``
    None) keeps its score."""
    if rating is None:
        return score
    return round(max(0.0, min(100.0, score * (1 + weight * (rating - 5) / 5))), 1)


def prior_group(base: str) -> str:
    """The style group of the vision rating prior (#948) for a style whose built-in base is
    ``base``: ``"action"`` for :data:`ACTION_STYLES`, ``"other"`` for every other style."""
    return "action" if base in ACTION_STYLES else "other"


def rating_centres(ratings: Iterable[tuple[str, float]], group: str) -> dict[str, float]:
    """The centre of each prompt kind's ratings (#909): ``ratings`` are ``(kind, rating)``
    pairs, the kind ``"speech"`` or ``"frames"``. A kind's centre is the mean of its ratings,
    shrunk toward the ``group``'s (:func:`prior_group`) prior in
    :data:`cutpoint.vision.RATING_PRIOR` while it has fewer than
    :data:`cutpoint.vision.PRIOR_N`: ``(sum + max(0, PRIOR_N - n) * prior) / max(n, PRIOR_N)``.
    Only the kinds rated are in the result."""
    sums: dict[str, list[float]] = {}
    for kind, r in ratings:
        sums.setdefault(kind, []).append(r)
    return {
        kind: (sum(rs) + max(0, PRIOR_N - len(rs)) * RATING_PRIOR[group, kind])
        / max(len(rs), PRIOR_N)
        for kind, rs in sums.items()
    }


def centred_rating(rating: float, centre: float) -> float:
    """``rating`` moved so its kind's ``centre`` (:func:`rating_centres`) is 5, within 0-10."""
    return max(0.0, min(10.0, 5 + rating - centre))


def sync_vision(c: dict[str, Any]) -> float | None:
    """Keep the vision rating of the candidate ``c`` only while it describes the clip (#802).

    ``vision_rated`` (written by :func:`_rate_vision`) holds the span the frames were rated
    over and the analysis score before and after the blend. While ``c`` has that span (within
    :data:`VISION_SPAN_EPS`) the rating counts; once the span changes (widened, trimmed, merged
    back) it moves to ``vision_stale`` (``score``, ``label``, ``window``): ``signals.vision``
    becomes None and ``vision_label`` goes. The span coming back restores both. The clip's
    ``label`` is never touched. Returns the analysis score to rank by (the
    blended one or the one before the blend); None for a candidate without ``vision_rated``
    (a project rated before this, or never rated), which is left as it is."""
    rated = c.get("vision_rated")
    if not isinstance(rated, dict):
        return None
    sig = c.setdefault("signals", {})
    stale = c.get("vision_stale")
    if (
        abs(c["start"] - rated["start"]) <= VISION_SPAN_EPS
        and abs(c["end"] - rated["end"]) <= VISION_SPAN_EPS
    ):
        if isinstance(stale, dict):
            sig["vision"] = stale.get("score")
            if stale.get("label"):
                c["vision_label"] = stale["label"]
            del c["vision_stale"]
        return rated["blended"]
    if not isinstance(stale, dict):
        stale = c["vision_stale"] = {
            "score": sig.get("vision"),
            "label": c.pop("vision_label", None),
            "window": [rated["start"], rated["end"]],
        }
        sig["vision"] = None
    return rated["base"]


def vision_skipped(c: dict[str, Any]) -> bool:
    """Whether the candidate ``c`` is a moment the default edit skipped above its cut when the
    vision step ran (too long or overlapping a pick; ``vision_skipped``, the span it had then)
    and still has that span (within :data:`VISION_SPAN_EPS`). The vision clamp
    (:func:`cutpoint.scorer.apply_vision_clamp`) does not count such a moment as a neighbour.
    A re-rank (:func:`cutpoint.scorer.rescore_events`) also removes the mark once the default
    edit of the ranking without the ratings takes the moment (#949)."""
    span = c.get("vision_skipped")
    return (
        isinstance(span, list)
        and abs(c["start"] - span[0]) <= VISION_SPAN_EPS
        and abs(c["end"] - span[1]) <= VISION_SPAN_EPS
    )


def _label(c: Candidate, style: Style) -> str:
    if c.label_hint:
        return slugify(c.label_hint, 30)
    if c.signals.get("laughter", 0) > 0.5:
        return "laugh"
    if c.signals.get("loudness", 0) > 0.6:
        return "loud-moment"
    return f"{style.name}-moment"


# --------------------------------------------------------------------------- main entry
@dataclass
class PartSignals:
    """What :func:`pick_candidates` needs of one analysed part (part-relative time)."""

    n: int
    sig: dict[str, np.ndarray]
    kw_hits: list[tuple[float, float, str]]
    laugh_hits: list[tuple[float, float, str]]
    words: list[dict[str, Any]]
    segments: list[Segment]
    cuts: list[float]  # for snapping; with ``edge_cuts``, those found around edges so far
    scenes: bool = False  # scene detection is on (the scene curve follows the style)
    edge_cuts: EdgeCuts | None = None  # the cuts are detected around candidate edges only


def text_signals(ps: PartSignals, style: Style, name: str) -> None:
    """Fill the style-dependent transcript signals of ``ps``: the keyword and laughter hits
    and the ``speech`` and ``laughter`` curves (profanity gated against the loudness for
    styles with ``profanity_alone`` False). ``name``: the source's, for the log."""
    n, sig, words = ps.n, ps.sig, ps.words
    kw_hits = keyword_hits(words, style.keywords)
    laugh_hits = laughter_hits(words, flat_markers(ps.segments))
    if not style.profanity_alone:
        n_before = len(kw_hits)
        n_profane = sum(1 for *_, p in kw_hits if p in PROFANE)
        kw_hits = gate_profanity(kw_hits, laugh_hits, sig["loudness"])
        if n_profane:
            logger.info(
                "%s: ignored %d of %d profanity hits with no other cue",
                name,
                n_before - len(kw_hits),
                n_profane,
            )
    sig["speech"] = keyword_signal(n, kw_hits, style.keyword_cap)
    sig["laughter"] = _impulse_signal(n, [(t, w) for t, w, _ in laugh_hits])
    ps.kw_hits, ps.laugh_hits = kw_hits, laugh_hits


def part_signals(
    span: float,
    lufs_data: tuple[np.ndarray, np.ndarray] | None,
    segments: list[Segment],
    cuts: list[float] | None,
    style: Style,
    name: str,
    text: bool = True,
) -> PartSignals:
    """The signal grids of a part ``span`` seconds long: loudness from ``lufs_data`` (the cached
    times and momentary loudness; None without audio), the transcript ``segments``' signals
    (``text`` False leaves them for :func:`text_signals`) and, with ``cuts`` (None: no scene
    detection), the scene curve of a style that weighs it (:func:`scene_signal`); ``cuts`` are
    also kept, for snapping. ``name``: the source's, for the log."""
    n = max(1, int(span / DT) + 1)
    sig = {k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    if lufs_data is not None:
        times, lufs = lufs_data
        sig["loudness"] = np.interp(np.arange(n) * DT, times, loudness_signal(lufs))
    words = flat_words(segments)
    ps = PartSignals(n, sig, [], [], words, segments, list(cuts or []), cuts is not None)
    if text:
        text_signals(ps, style, name)
    sig["continuity"] = continuity_signal(n, words)
    scene_signal(ps, style)
    return ps


def scene_signal(ps: PartSignals, style: Style) -> None:
    """The scene curve of ``ps`` for ``style``: from the cuts when scene detection is on and the
    style weighs it (:func:`scores_scenes`), else zeros (it would add nothing to the score). Cuts
    found only around edges (``edge_cuts``) are replaced by a full pass first."""
    if not ps.scenes:
        return
    if not scores_scenes(style):
        ps.sig["scene"] = np.zeros(ps.n)
        return
    if ps.edge_cuts is not None:  # --style auto chose a style that weighs scenes after all
        ps.cuts, ps.edge_cuts = ps.edge_cuts.full(), None
    ps.sig["scene"] = _impulse_signal(ps.n, [(t, 0.7) for t in ps.cuts], sigma_s=1.5)


def _cover_edges(ps: PartSignals, cands: Sequence[Candidate]) -> None:
    """With ``ps.edge_cuts``, detect the scene cuts around the edges of ``cands`` (part-relative,
    not snapped yet) so :func:`_snap` finds every cut it can reach."""
    if ps.edge_cuts is not None and cands:
        ps.cuts = ps.edge_cuts.around(t for c in cands for t in (c.start, c.end))


@dataclass
class GridPart:
    """The grids of one analysed part, for :func:`nominate_candidates`: a :class:`SourceResult`
    of this run, or what the caches of an earlier one give (:func:`cached_grid_parts`)."""

    source: Path
    index: int  # of the source in highlights.json's ``sources``
    offset: float
    duration: float
    signals: PartSignals
    avail: frozenset[str] | None = None  # set by :func:`pick_candidates`


class PickPart(Protocol):
    """What :func:`pick_candidates` reads of a part, and its ``avail`` it sets: a
    :class:`SourceResult` of this run, or a :class:`GridPart` of :func:`cached_grid_parts`."""

    source: Path
    offset: float
    duration: float
    signals: PartSignals | None
    avail: frozenset[str] | None


@dataclass
class SourceResult:
    """Output of :func:`analyze_source` for one source or recovered part.

    Attributes:
        source: Source as given (original path; ``info`` probes its ``source_remap`` copy).
        info: Probe result of the file.
        offset: Absolute start of this part (0 for the file itself).
        duration: Seconds of this part that were actually analysed.
        candidates: Candidates in absolute source time.
        segments: Transcript in absolute source time.
        read_error: The ``read_errors.json`` record if reading was cut short.
        signals: The signals the candidates are picked from.
        avail: The features the candidates' scores were divided by (None until picked).
    """

    source: Path
    info: MediaInfo
    offset: float
    duration: float
    candidates: list[Candidate]
    segments: list[Segment]
    read_error: dict[str, Any] | None
    signals: PartSignals | None = None
    stages: dict[str, Any] = field(default_factory=dict)  # seconds per stage, timings.sources
    avail: frozenset[str] | None = None


def _shortfall_limit(expected: float) -> float:
    """Missing audio (s) that counts as truncation: ~2 s or 1%, whichever is less."""
    return max(SHORTFALL_FLOOR_S, min(SHORTFALL_S, SHORTFALL_FRAC * expected))


def check_read(
    src: Path, info: MediaInfo, offset: float, wav: Path, cdir: Path, res: ProgressResult | None
) -> dict[str, Any] | None:
    """Detect a read error from ffmpeg's messages and the extracted audio length.

    ffmpeg exits 0 after "Error during demuxing" / I/O errors and leaves a
    truncated WAV, so both signals are checked. With a cached WAV (``res`` is
    None) the record saved at extraction time is reused, or, for WAVs cached
    before this check existed, only the length is compared.

    Returns:
        The read-error record (also saved as ``read_errors.json`` in ``cdir``),
        or None if the audio is complete.
    """
    saved = load_read_errors(cdir)
    if res is None and saved is not None:
        return saved
    expected = info.duration - offset
    got = wav_duration(wav)
    short = expected - got
    messages = (
        list(dict.fromkeys(m.strip() for m in READ_ERROR_RE.findall(res.stderr)))[:5] if res else []
    )
    if not messages and short <= _shortfall_limit(expected):
        (cdir / READ_ERRORS_FILE).unlink(missing_ok=True)  # stale record from a bad read
        return None
    record = {
        "version": 1,
        "source": str(src),
        "offset": round(offset, 3),
        "container_duration": round(info.duration, 3),
        "expected_audio_s": round(expected, 3),
        "audio_s": round(got, 3),
        "missing_s": round(max(0.0, short), 3),
        "failure_time": round(offset + got, 3),
        "messages": messages,
        "detected_by": "ffmpeg messages + length" if res else "length check (cached audio)",
        "resumed_at": None,
    }
    cache.save_json(cdir / READ_ERRORS_FILE, record)
    return record


def warn_read_error(src: Path, rec: dict[str, Any], recover: bool) -> None:
    """Print a loud, unmissable warning about a truncated read."""
    t = rec["failure_time"]
    bar = "!" * 78
    msgs = "; ".join(rec["messages"]) or "no ffmpeg message (audio shorter than container)"
    hint = (
        "--recover: probing for a readable point after it"
        if recover
        else "re-run with --recover to analyse the footage after the bad region"
    )
    logger.warning(bar)
    logger.warning("READ ERROR in %s", src)
    logger.warning(
        "  reading stopped at ~%s (%.1f s); %.1f s of %.1f s (%.0f%%) NOT analysed",
        clock(t),
        t,
        rec["missing_s"],
        rec["expected_audio_s"],
        100 * rec["missing_s"] / max(rec["expected_audio_s"], 1e-6),
    )
    logger.warning("  ffmpeg: %s", msgs)
    logger.warning("  %s", hint)
    logger.warning(bar)


def find_resume_point(src: Path, fail_t: float, duration: float) -> float | None:
    """First readable time after ``fail_t``, found with a few small probe reads.

    Probes at growing distances (5, 10, 20, ... s) and stops at the first good
    read, then bisects ``RESUME_REFINE`` times toward the failure to lose less
    footage. Each probe decodes only ~3 s, and a failed spot is never retried.

    Returns:
        Absolute resume time, or None if nothing readable was found.
    """
    last_bad, good = fail_t, None
    for step in RESUME_STEPS_S:
        t = fail_t + step
        if t >= duration - 5:
            break
        ok = read_window_ok(src, t)
        logger.info("  recover probe %s: %s", clock(t), "readable" if ok else "read error")
        if ok:
            good = t
            break
        last_bad = t
    if good is None:
        return None
    for _ in range(RESUME_REFINE):
        if good - last_bad < 2.0:
            break
        mid = (last_bad + good) / 2
        ok = read_window_ok(src, mid)
        logger.info("  recover probe %s: %s", clock(mid), "readable" if ok else "read error")
        if ok:
            good = mid
        else:
            last_bad = mid
    return round(min(good + RESUME_MARGIN_S, duration), 3)


def _shift_segments(segments: list[Segment], dt: float) -> list[Segment]:
    if not dt:
        return segments
    out = []
    for seg in segments:
        words = [
            {**w, "start": w["start"] + dt, "end": w["end"] + dt} for w in seg.get("words", [])
        ]
        shifted = {**seg, "start": seg["start"] + dt, "end": seg["end"] + dt, "words": words}
        if seg.get("markers"):
            shifted["markers"] = [
                {**m, "start": m["start"] + dt, "end": m["end"] + dt} for m in seg["markers"]
            ]
        out.append(shifted)
    return out


def analyze_source(
    src: Path,
    style: Style,
    tr: Transcriber,
    use_scenes: bool,
    max_candidates: int | None,
    timings: dict[str, float],
    offset: float = 0.0,
    recover: bool = False,
    cam_ctx: Any = None,
    pick: bool = True,
    report: Callable[[float], None] | None = None,
    read_checked: Callable[[MediaInfo, dict[str, Any] | None], Any] | None = None,
    text: bool = True,
    on_transcript: Callable[[], None] | None = None,
    cpu_busy: Callable[[], bool] | None = None,
) -> SourceResult:
    """Compute signals for one source (or a recovered part starting at ``offset``).

    Internally everything runs in part-relative time; candidates and the
    transcript are returned in absolute source time. With ``cam_ctx`` (a
    :class:`cutpoint.actioncam.ActionCamContext`), candidates come from the action-cam
    feature grid instead of the generic signals. ``src`` is the original path
    (recorded everywhere); files are read through ``source_remap``. With ``pick``
    False no candidates are picked yet (:func:`pick_candidates` does it later); with
    ``text`` False neither are the style's transcript signals (:func:`text_signals`, for a
    style decided after transcription; ``pick`` must then be False too).
    ``report`` gets the share of the source's transcription done (0-1): the audio
    extract is its first :data:`EXTRACT_SHARE`, Whisper the rest. ``read_checked`` gets
    the probe and the read-error record (None: a clean read) as soon as they are known,
    before loudness, Whisper and the scene cuts; what it returns goes to
    :func:`scene_cuts` as ``shared`` (a scene proxy made in the background, or None).
    ``on_transcript`` is called once the transcript is there (made or cached), before the
    scene cuts. ``cpu_busy`` says whether a CPU-heavy step of the caller (the background HUD
    OCR) runs right now (None: none does); the scene cuts split their decode between NVDEC and
    the CPU only when it does not, nor Whisper on the CPU (:func:`_hybrid_plan`).
    """

    stages: dict[str, Any] = {}

    def add(name: str, took: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + took, 2)
        stages[name] = round(stages.get(name, 0.0) + took, 2)

    def tick(name: str, t0: float) -> None:
        add(name, time.perf_counter() - t0)

    io = remap_source(src)  # read the copy, record the original
    t0 = time.perf_counter()
    info = probe(io)
    stages["probe_s"] = round(time.perf_counter() - t0, 2)
    cdir = cache.source_dir(src, offset)
    span = info.duration - offset
    read_error: dict[str, Any] | None = None
    shared = None  # read_checked's background scene proxy
    overlap: _SceneJob | None = None  # the scene cuts, run alongside Whisper
    segments: list[Segment] = []
    lufs_data: tuple[np.ndarray, np.ndarray] | None = None
    if info.audio_streams:
        t0 = time.perf_counter()
        wav = cdir / "audio16k.wav"
        res = None

        def at(frac: float) -> float:
            """Position of ``frac`` of this part in the whole source (0-1)."""
            return (offset + frac * span) / info.duration

        def extracted(frac: float) -> None:
            if report is not None:
                report(EXTRACT_SHARE * at(frac))

        def transcribed(frac: float) -> None:
            if report is not None:
                report(EXTRACT_SHARE + (1 - EXTRACT_SHARE) * at(frac))

        tracks = _analysis_tracks(src, io, info, cdir)
        made = not wav.exists() or cached_transcript(tr, cdir) is None
        if not wav.exists():
            with _watch_extract(wav, span, extracted if report else None):
                res = extract_audio(io, wav, info, start=offset, tracks=tracks)
        tick("audio_extract_s", t0)
        read_error = check_read(src, info, offset, wav, cdir, res)
        if read_error:
            warn_read_error(src, read_error, recover)
            span = read_error["audio_s"]  # analyse only what was actually read
        if read_checked is not None:
            shared = read_checked(info, read_error)

        t0 = time.perf_counter()
        lufs_data = _loudness(wav, cdir)
        tick("loudness_s", t0)

        t0 = time.perf_counter()
        try:
            if (
                use_scenes
                and not edge_windows(style, span)
                and scene_overlap()
                and shared is None
                and cache.load_json(cdir / SCENES_FILE) is None
                and cached_transcript(tr, cdir) is None
            ):
                tr.load()  # the memory guard runs before the scene thread takes any VRAM
                # Whisper runs meanwhile: the split decode only while it is on the GPU
                free = getattr(tr, "device", None) == "cuda" and not (cpu_busy and cpu_busy())
                limit = span if read_error else None
                overlap = _SceneJob(io, cdir, span, offset, limit, shared, free)
            segments = _transcript(wav, cdir, tr, transcribed if report else None, src)
            tick("transcribe_s", t0)
            if on_transcript is not None:
                on_transcript()
            if overlap is not None:
                overlap.result()  # raises what the scene detection raised
        except BaseException:  # Whisper failed or was stopped: so are the scene cuts
            if overlap is not None:
                overlap.stop()
            raise
        segments, dropped = filter_hallucinations(segments)
        segments = split_markers(segments)  # non-speech markers: only the laughter hits see them
        if dropped:
            logger.info("%s: dropped %d likely-hallucinated transcript segments", src.name, dropped)
        stages["cached"] = not made
    else:
        stages["cached"] = False
        logger.warning("%s has no audio; using scene/vision signals only", src.name)
        if read_checked is not None:
            shared = read_checked(info, None)

    cuts: list[float] = []
    edge_cuts: EdgeCuts | None = None
    # A style that does not weigh the scene curve: cuts only around candidate edges (#730), in a
    # container that seeks exactly (MPEG-TS takes the full pass, #976)
    windowed = use_scenes and edge_windows(style, span) and _seeks_exactly(io)
    if windowed:  # a full pass already cached is used as it is
        if (found := cache.load_json(cdir / SCENES_FILE)) is not None:
            cuts = found
        else:
            stages["scene_windows"] = True
            edge_cuts = EdgeCuts(
                io,
                cdir,
                span,
                offset,
                span if read_error else None,
                on_time=lambda took: add("scenes_s", took),
            )
    elif use_scenes:
        t0 = time.perf_counter()
        if overlap is not None:
            cuts = overlap.cuts
            took = overlap.took  # the thread's own time, mostly inside the transcription
            stages["overlap"] = True
        else:
            free = {} if cpu_busy is not None and cpu_busy() else {"cpu_free": True}
            cuts = scene_cuts(
                io, cdir, span, offset, limit=span if read_error else None, shared=shared, **free
            )
            took = time.perf_counter() - t0
        add("scenes_s", took)
        if shared is not None:  # for measuring the shared decode
            logger.info(
                "%s: shared decode (%s): scenes %.1fs, %s",
                src.name,
                shared.mode,
                took,
                shared.summary(),
            )
    ps = part_signals(
        span, lufs_data, segments, cuts if use_scenes else None, style, src.name, text
    )
    ps.edge_cuts = edge_cuts

    res = SourceResult(
        source=src,
        info=info,
        offset=offset,
        duration=span,
        candidates=[],
        segments=_shift_segments(segments, offset),
        read_error=read_error,
        signals=ps,
        stages=stages,
    )
    if pick:
        res.candidates = pick_candidates(res, style, max_candidates, timings, cam_ctx)
    where = f" from {clock(offset)}" if offset else ""
    logger.info(
        "%s%s: %.0fs, %d words, %d scene cuts%s%s",
        src.name,
        where,
        span,
        len(ps.words),
        len(ps.cuts),
        " (around candidate edges)" if ps.edge_cuts is not None else "",
        f", {len(res.candidates)} candidates" if pick else "",
    )
    return res


def pick_candidates(
    res: PickPart,
    style: Style,
    max_candidates: int | None,
    timings: dict[str, float],
    cam_ctx: Any = None,
) -> list[Candidate]:
    """Candidates of one analysed part (:func:`analyze_source`, or a :class:`GridPart`), in
    absolute source time."""
    src, span, offset = res.source, res.duration, res.offset
    ps = res.signals
    if ps is None:
        raise ValueError(f"{src}: no signals to pick candidates from")
    n, sig = ps.n, ps.sig
    segments = ps.segments
    avail = scorer.available_for(style)
    if cam_ctx is not None:
        cands, avail = _action_cam_candidates(
            src, n, span, sig, offset, style, cam_ctx, max_candidates, segments, timings, avail
        )
    else:
        usual = usual_count(style) if style.open_list and max_candidates is None else None
        # Picked on the raw weighted sum: the cut-offs do not depend on what is available.
        combined = scorer.combined(sig, style)
        raw = _pick_windows(
            src, combined, sig, style, span, candidate_cap(style, max_candidates), usual
        )
        # The first ``usual`` picks are finished as they always were; the picks past them only
        # fill what those leave free, so a longer list never changes the top of a shorter one.
        head, tail = (raw[:usual], raw[usual:]) if usual is not None else (raw, [])
        cands = _merge(head, style)
        _cover_edges(ps, cands)
        for c in cands:
            _snap(c, ps.cuts, segments, span, style)
        cands = _resolve_overlaps(cands, style)
        taken = [(c.start, c.end) for c in cands]
        tail = [c for c in tail if not any(a <= c.peak_t <= b for a, b in taken)]
        cands += _place_free(tail, taken, ps, span, style)
    res.avail = avail
    return describe_candidates(cands, style, ps, offset, avail)


DIALOGUE_PRE_ROLL_S = 1.0  # a dialogue candidate starts this long before its stretch


def usual_count(style: Style) -> int:
    """The candidates a part gets by default: enough to cover ~3x the target duration, at least
    30. A style with an open list (``Style.open_list``) goes past it, above
    :data:`LONG_LIST_FLOOR`; up to it the floor is :data:`PICK_FLOOR`."""
    return max(30, int(3 * style.target_total / ((style.min_len + style.max_len) / 2)))


def candidate_cap(style: Style, explicit: int | None = None) -> int:
    """Most candidates picked from a part (before merging): ``explicit`` (``--max-candidates``,
    with the :data:`PICK_FLOOR` throughout) if given, else :func:`usual_count`, or for an open
    list as many as qualify (:func:`_pick_windows`: :data:`PICK_FLOOR` up to
    :func:`usual_count`, then :data:`LONG_LIST_FLOOR`)."""
    if explicit is not None:
        return explicit
    return sys.maxsize if style.open_list else usual_count(style)


def describe_candidates(
    cands: list[Candidate],
    style: Style,
    ps: PartSignals,
    offset: float,
    avail: frozenset[str] | None = None,
) -> list[Candidate]:
    """Score ``cands`` from their peak and fill their keywords, label hint and transcript
    snippet from the part's transcript; then move them to absolute source time (``offset``).
    ``avail``: the features the score is renormalised over (default
    :func:`cutpoint.scorer.available_for`; an action-cam source's own, #950)."""
    words, kw_hits, laugh_hits = ps.words, ps.kw_hits, ps.laugh_hits
    if avail is None:
        avail = scorer.available_for(style)
    for c in cands:
        c.score = scorer.score(c.peak, style, avail, c.must_include)
        inside = [(t, w, p) for t, w, p in kw_hits + laugh_hits if c.start <= t <= c.end]
        c.keywords = [p for _, _, p in sorted(inside, key=lambda h: -h[1])]
        c.keywords = list(dict.fromkeys(c.keywords))[:5]
        strong = [
            p for _, w, p in sorted(inside, key=lambda h: -h[1]) if w >= 0.5 and p not in PROFANE
        ]  # a bare swear is a useless label
        if not c.label_hint:  # action-cam picks bring their own ("jump 0.6s", "crash 7g")
            c.label_hint = strong[0] if strong else ""
        said = [w for w in words if c.start <= w["start"] <= c.end]
        c.transcript, c.transcript_unsure = _transcript_snippet(said)
        if offset:  # back to absolute source time: clips cut the original file
            c.start, c.end = round(c.start + offset, 3), round(c.end + offset, 3)
            c.peak_t += offset
    return cands


def _free_around(t: float, taken: list[tuple[float, float]], span: float) -> tuple[float, float]:
    """The stretch of ``[0, span]`` around ``t`` that no ``taken`` window covers."""
    if any(a < t < b for a, b in taken):
        return t, t
    lo = max([0.0, *(b for a, b in taken if b <= t)])
    hi = min([span, *(a for a, b in taken if a >= t)])
    return lo, hi


def _fit_free(c: Candidate, taken: list[tuple[float, float]], span: float, style: Style) -> bool:
    """Keep ``c`` out of the ``taken`` windows (existing candidates), at least ``min_len``
    long: moved within the free stretch around its event; False if that is too short."""
    lo, hi = _free_around(c.peak_t, taken, span)
    c.start, c.end = max(c.start, lo), min(c.end, hi)
    if c.end - c.start < style.min_len:
        c.end = min(hi, c.start + style.min_len)
        c.start = max(lo, c.end - style.min_len)
    return c.end - c.start >= style.min_len - 1e-6


def _place_free(
    cands: list[Candidate],
    taken: list[tuple[float, float]],
    ps: PartSignals,
    span: float,
    style: Style,
) -> list[Candidate]:
    """Finish ``cands`` (part-relative windows) like the picked ones, in what the ``taken``
    windows leave free: fitted, merged, snapped, overlaps resolved and fitted again; none
    ends up over a taken window or another."""
    stretches: dict[tuple[float, float], list[Candidate]] = {}
    for c in cands:
        if c.over or _fit_free(c, taken, span, style):
            stretches.setdefault(_free_around(c.peak_t, taken, span), []).append(c)
    cands = [m for group in stretches.values() for m in _merge(group, style)]
    _cover_edges(ps, cands)
    for c in cands:
        _snap(c, ps.cuts, ps.segments, span, style)
    cands = [
        c for c in _resolve_overlaps(cands, style) if c.over or _fit_free(c, taken, span, style)
    ]
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: c.start):  # the fits can bring two together again
        if out and c.start < out[-1].end:
            c.start = out[-1].end
            if c.end - c.start < style.min_len:
                continue
        out.append(c)
    return out


def nominate_candidates(
    part: GridPart, style: Style, seeds: list[Seed], taken: list[tuple[float, float]]
) -> list[Candidate]:
    """Candidates for the events ``seeds`` (absolute source time) of one analysed part that no
    existing candidate covers, in absolute source time and scored from the part's grids.

    A window runs from ``pre_roll`` before an event to its tail (the event type's ``tail_s``,
    else ``post_roll``) after it, at least ``min_len`` long. Windows close to each other merge
    (:func:`_merge`: a burst of kills is one candidate) and are snapped like the picked ones.
    All stay out of ``taken`` (the windows of the existing candidates, absolute time), and
    ``style.max_len`` holds as it does for the picked ones. A dialogue stretch (``Seed.whole``)
    is the exception: its window is the span the dialogue reader marked, :data:`DIALOGUE_PRE_ROLL_S`
    before it, never cut to ``max_len`` or merged into another window (it can run for minutes).
    A stretch with ``Seed.over`` also ignores ``taken``: it stays whole over the candidates it overlaps.
    """
    ps, off, span = part.signals, part.offset, part.duration
    taken = [(a - off, b - off) for a, b in taken]
    cands = []
    for s in sorted(seeds, key=lambda s: s.start):
        t = s.start - off
        if not 0.0 <= t <= span:
            continue
        if s.whole:  # a dialogue stretch: as the reader marked it, whatever its length
            start, end = max(0.0, t - DIALOGUE_PRE_ROLL_S), min(span, t + s.hold)
        else:
            end = t + s.hold + (style.post_roll if s.tail_s is None else s.tail_s)
            start, end = max(0.0, t - style.pre_roll), min(span, end)
            if end - start > style.max_len:
                end = start + style.max_len
        if end - start < style.min_len:
            end = start + style.min_len
        cands.append(
            Candidate(
                part.source,
                start,
                end,
                t,
                0.0,
                dict.fromkeys(ps.sig, 0.0),
                label_hint=slugify(s.name, 30),
                nominated=[s.name],
                whole=s.whole,
                over=s.over,
            )
        )
    out = _place_free(cands, taken, ps, span, style)
    combined = scorer.combined(ps.sig, style)
    for c in out:
        a = int(c.start / DT)
        b = max(a + 1, int(c.end / DT))
        c.peak = float(combined[a:b].max())
        c.signals = {k: round(float(v[a:b].max()), 3) for k, v in ps.sig.items()}
    return describe_candidates(out, style, ps, off)


def cached_grid_parts(hl: dict[str, Any], style: Style) -> list[GridPart]:
    """The grids of the analysis ``hl`` as its caches give them (what :func:`analyze_source`
    built: the loudness, the transcript of the analysis's model and language, the scene cuts).
    A source with a recovered part, or offline, has none; nor has one whose caches are gone,
    unless the style detects its scene cuts around candidate edges (:func:`edge_windows`): such
    a source (also one without audio, so without loudness or transcript) gets its part, the
    cuts decoded when an edge needs them. Otherwise a source without :data:`SCENES_FILE` gets no
    scene curve or cuts (with a style that weighs the curve, an INFO line names it;
    ``analyze`` detects them again)."""
    settings = hl.get("settings") or {}
    entries = hl.get("sources", [])
    recovered = {e["path"] for e in entries if e.get("offset") or e.get("read_errors")}
    out: list[GridPart] = []
    for i, e in enumerate(entries):
        if e["path"] in recovered:
            continue
        src = Path(e["path"])
        try:
            cdir = cache.source_dir(src)
        except OSError:
            continue
        loud = cache.load_json(cdir / "loudness.json")
        segments = source_segments(src, settings.get("language"), settings.get("whisper_model"))
        found = cache.load_json(cdir / SCENES_FILE)
        scenes = bool(settings.get("scenes", True))
        span = float(e.get("analyzed_s") or e["duration"])
        windowed = scenes and found is None and edge_windows(style, span)  # EdgeCuts serve it
        if loud is None and segments is None and found is None and not windowed:
            continue  # nothing cached to build from
        if scenes and found is None and scores_scenes(style):
            logger.info(
                "%s: no scene cuts cached; its scene score is missing until analyze runs again",
                src.name,
            )
        cuts = (found or []) if scenes else None
        lufs = (np.asarray(loud["t"]) - 0.2, np.asarray(loud["m"])) if loud else None
        ps = part_signals(span, lufs, segments or [], cuts, style, src.name)
        if windowed:  # as the analysis did: cuts around edges (scenes-windows.json, else decoded)
            ps.edge_cuts = EdgeCuts(remap_source(src), cdir, span)
        out.append(GridPart(src, i, 0.0, span, ps))
    return out


def _transcript_snippet(
    words: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, float]] | None]:
    """A transcript snippet of ``words`` (at most ``SNIPPET_CHARS``) and the words Whisper was
    unsure of: ``(index, probability)`` for each word of ``snippet.split(" ")`` (the last one
    may be cut short) below ``UNSURE_P``. The unsure list is None when no word is unsure or a
    word has no ``probability`` (transcripts cached before it was stored). Words are split at
    whitespace, so a word with an inner space counts as several with its probability and an
    empty one is skipped."""
    toks = [(part, w.get("probability")) for w in words for part in w["word"].split()]
    text = " ".join(t for t, _ in toks)[:SNIPPET_CHARS].rstrip()
    probs = [p for _, p in toks[: len(text.split(" ")) if text else 0]]
    if any(p is None for p in probs):
        return text, None
    return text, [(i, p) for i, p in enumerate(probs) if p < UNSURE_P] or None


def _action_cam_candidates(
    src: Path,
    n: int,
    span: float,
    sig: dict[str, np.ndarray],
    offset: float,
    style: Style,
    cam_ctx: Any,
    max_candidates: int | None,
    segments: list[Segment],
    timings: dict[str, float],
    avail: frozenset[str] | None,
) -> tuple[list[Candidate], frozenset[str] | None]:
    """Candidates from the action-cam feature grid (telemetry + audio + VLM tags), and the
    features their score is renormalised over: the project's ``avail``, without the telemetry
    features for a source without telemetry, or without ``speed`` for one whose telemetry has
    no GPS rows and no wind reading (#959); the pick floor of either is scaled to match (#950)."""
    from cutpoint import actioncam  # noqa: PLC0415

    t0 = time.perf_counter()
    feats, mask, friends = actioncam.build_features(src, n, sig, cam_ctx, offset)
    tel = cam_ctx.telemetry.get(str(src))
    events = tel.events if tel is not None else []
    src_avail = actioncam.source_available(avail, tel)
    actioncam.save_features(cam_ctx.project, src, offset, span, feats, mask, friends, events)
    if max_candidates is None:  # short clips: a few; long ones: one per ~20 s
        max_candidates = max(4, int(span / 20))
    picks = actioncam.pick(
        feats,
        style.weights,
        mask,
        friends,
        events,
        span,
        max_candidates,
        offset,
        friends_strong=style.mask_friends,
        floor=actioncam.source_floor(style.weights, avail, src_avail),
    )
    forced = actioncam.must_include_picks(src, cam_ctx, style.must_include_crash, span, offset)
    if forced:
        logger.info(
            "%s: %d must-include crash clip(s) (confidence >= %.2f)",
            src.name,
            len(forced),
            style.must_include_crash,
        )
        picks = actioncam.merge_must_include(picks, forced)
    cands = []
    for p in picks:
        c = Candidate(
            source=src,
            start=p.start,
            end=p.end,
            peak_t=p.peak_t,
            peak=p.peak,
            signals=dict(p.signals),
            label_hint=p.label,
            kind=p.kind,
            must_include=any(p is f for f in forced),
            event=p.event if any(p is f for f in forced) else None,
        )
        # Word edges only (no scene snapping): never cut a word in half.
        _snap(c, [], segments, span, style)
        cands.append(c)
    timings["action_cam_s"] = round(timings.get("action_cam_s", 0.0) + time.perf_counter() - t0, 2)
    return cands, src_avail


CRASH_KIND = (
    "crash"  # the ``kind`` of a crash moment outside action-cam (action-cam's: ``wipeout``)
)


def _crash_event(c: dict[str, Any]) -> dict[str, Any]:
    """The ``crash`` record of a candidate for the crashes.json crash ``c`` (as action-cam's)."""
    return {
        "type": "crash",
        "id": c.get("id"),
        "t": c["t0"],
        "end": c["t1"],
        "confidence": c["confidence"],
    }


def _crash_entries(crashes: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The usable crashes of a crashes.json object: a source, a finite confidence, ``t0`` and
    ``t1``, and a planned clip (with a reaction that has a start and end, or none)."""

    def num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    raw = crashes.get("crashes") if isinstance(crashes, dict) else None
    return [
        c
        for c in (raw if isinstance(raw, list) else [])
        if isinstance(c, dict)
        and isinstance(c.get("source"), str)
        and all(num(c.get(k)) for k in ("confidence", "t0", "t1"))
        and isinstance(c.get("clip"), dict)
        and num(c["clip"].get("start"))
        and num(c["clip"].get("end"))
        and (
            (r := c["clip"].get("reaction")) is None
            or (isinstance(r, dict) and num(r.get("start")) and num(r.get("end")))
        )
    ]


class CrashPart(Protocol):
    """What :func:`add_crash_candidates` reads and changes of a part: a :class:`SourceResult`
    of this run, or a :class:`HighlightsPart` of an earlier one."""

    source: Path
    offset: float
    duration: float
    candidates: list[Candidate]
    signals: PartSignals | None


@dataclass
class HighlightsPart:
    """A part of an earlier analysis, for :func:`rebuild_crash_moments`: its cached grids and
    the candidates of its highlights.json records."""

    source: Path
    offset: float
    duration: float
    candidates: list[Candidate]
    signals: PartSignals | None


def _crash_windows(r: CrashPart, c: dict[str, Any], style: Style, forced: bool) -> list[Candidate]:
    """The candidates of the crash ``c`` in the part ``r`` (absolute source time): its planned
    clip (approach, impact, aftermath) and, when ``forced``, its reaction. A forced one scores
    its confidence (:func:`cutpoint.scorer.score`); any other is scored from the part's grids
    over its span, as a picked window there would be. Word edges only (no scene snapping)."""
    ps, off, span = r.signals, r.offset, r.duration
    if ps is None:
        return []
    spans = [("crash", c["clip"]["start"], c["clip"]["end"], c["t0"])]
    reaction = c["clip"].get("reaction")
    if forced and isinstance(reaction, dict):
        spans.append(("crash reaction", reaction["start"], reaction["end"], reaction["start"]))
    words_only = replace(style, snap_sentences=False)
    combined = scorer.combined(ps.sig, style)
    out = []
    for label, a, b, peak_t in spans:
        a, b = max(0.0, a - off), min(span, b - off)
        if b - a < 1.0:
            continue  # in another part of a recovered source
        cand = Candidate(
            r.source,
            a,
            b,
            peak_t - off,
            0.0,
            {},
            label_hint=label,
            kind=CRASH_KIND,
            must_include=forced,
            event=_crash_event(c),
        )
        _snap(cand, [], ps.segments, span, words_only)
        i = max(0, min(int(cand.start / DT), ps.n - 1))
        j = max(i + 1, int(cand.end / DT))
        cand.signals = {k: round(float(v[i:j].max()), 3) for k, v in ps.sig.items()}
        cand.peak = float(c["confidence"]) if forced else float(combined[i:j].max())
        out.append(cand)
    return describe_candidates(out, style, ps, off)


def add_crash_candidates(
    results: Sequence[CrashPart], style: Style, crashes: dict[str, Any] | None
) -> None:
    """Make the crashes of ``crashes`` (a crashes.json object) moments of ``results`` (any
    style but action-cam, whose own scorer reads crashes.json).

    Only crashes at or above the project's floor (:func:`cutpoint.crash.crash_floor`) count.
    One at or above ``style.must_include_crash`` (0: none) is forced, as in action-cam: its
    clip and its reaction become must-include candidates (``kind`` crash) that replace the
    candidates they overlap. A weaker one marks the candidate it overlaps most (its ``event``;
    span and score stay), or else becomes a candidate of its own, scored and ranked like any
    other. A crash counts in the part (of a recovered source) its ``t0`` falls in."""
    from cutpoint.crash import crash_floor  # noqa: PLC0415

    entries = _crash_entries(crashes)
    if not entries:
        return
    floor = crash_floor(c["confidence"] for c in entries)
    at = style.must_include_crash
    n_forced = n_new = n_marked = 0
    for r in results:
        if r.signals is None:
            continue
        key = path_key(str(r.source))
        mine = [
            c
            for c in entries
            if c["confidence"] >= floor
            and path_key(c["source"]) == key
            and r.offset <= c["t0"] < r.offset + r.duration
        ]
        forced = [c for c in mine if at > 0 and c["confidence"] >= at]
        made = [x for c in forced for x in _crash_windows(r, c, style, forced=True)]
        if made:
            r.candidates = made + [
                x
                for x in r.candidates
                if not any(x.start < f.end and f.start < x.end for f in made)
            ]
            n_forced += len(forced)
        weak = sorted((c for c in mine if c not in forced), key=lambda c: -c["confidence"])
        for c in weak:
            a, b = c["clip"]["start"], c["clip"]["end"]
            hits = [x for x in r.candidates if x.start < b and a < x.end]
            if hits:
                host = max(hits, key=lambda x: min(b, x.end) - max(a, x.start))
                if host.event is None:  # a crash moment shows its own crash
                    host.event = _crash_event(c)
                    n_marked += 1
                continue
            new = _crash_windows(r, c, style, forced=False)
            r.candidates += new
            n_new += len(new)
    if n_forced or n_new or n_marked:
        logger.info(
            "Crashes: %d must-include (confidence >= %.2f), %d new moment(s), %d marked "
            "(floor %.2f)",
            n_forced,
            at,
            n_new,
            n_marked,
            floor,
        )


CRASH_MARKS = ("crash", "crash_confidence", "crash_forced")  # the keys of a candidate's crash


def _crash_keys(recs: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """The crash moments ``recs`` by what names one across runs: its part and crash (id and
    impact span, not its confidence) and its place among that crash's windows in time order
    (0: its clip, 1: its reaction). Not its span, which menu time may have trimmed, nor its
    label."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rec in sorted(recs, key=lambda r: (r["source_index"], r["start"])):
        crash = rec.get("crash") or {}
        base = (rec["source_index"], crash.get("id"), crash.get("t"), crash.get("end"))
        out[(*base, sum(k[:4] == base for k in out))] = rec
    return out


def _canonical(hl: dict[str, Any]) -> dict[str, Any]:
    """``hl`` with its candidates in a fixed order, to compare two analyses whatever the order."""
    cands = sorted(hl.get("candidates", []), key=lambda c: str(c.get("id", "")))
    return {**hl, "candidates": cands}


def rebuild_crash_moments(project: Path) -> bool:
    """Rebuild the crash moments of ``<project>/highlights.json`` from the current crashes.json,
    as :func:`analyze` and :func:`reanalyze` build them (:func:`add_crash_candidates`), for a
    re-run of ``cutpoint events`` that does not re-analyse. Returns whether it wrote the file.

    The crash moments and marks of the analysis are taken off and made again over the grids its
    caches give (:func:`cached_grid_parts`), so a moment whose crash is unchanged stays as it is
    (id, vision rating, a span menu time trimmed) and repeat runs change nothing. A moment is
    matched by its crash, not its span. As in :func:`analyze`, the candidates events nominated
    host no crash (they come after the crash moments; the re-rank nominates again) and menu time
    drops or trims a new crash moment (:func:`drop_menu_time`) before it takes an id. New ones
    get the next ids (``settings.last_candidate_id``). Left alone: action-cam (its scorer reads
    crashes.json), 360 projects, ``settings.crash_moments`` false, and a source whose grids are
    not cached (recovered, offline). A candidate that a forced crash moment replaced is not brought back
    when that crash goes away: only a re-analysis does.

    Raises:
        ValueError: If the style is not in the library and the file records no built-in base
            (:func:`_recorded_style`).
    """
    from cutpoint.crash import load_crashes  # noqa: PLC0415
    from cutpoint.pano.project import PANO_FILE  # noqa: PLC0415
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415

    path = project / "highlights.json"
    hl = cache.load_json(path)
    if not isinstance(hl, dict) or "style" not in hl or "sources" not in hl:
        return False
    if (hl.get("settings") or {}).get("crash_moments") is False or (project / PANO_FILE).is_file():
        return False
    style = _recorded_style(hl)
    if style.kind == ACTION_CAM:
        return False
    parts = {
        p.index: HighlightsPart(p.source, p.offset, p.duration, [], p.signals)
        for p in cached_grid_parts(hl, style)
    }
    if not parts:
        return False
    cands = hl.get("candidates", [])
    restore_widened(cands)  # as analyze made the crash moments, before any widening
    before = copy.deepcopy(hl)
    rest: list[dict[str, Any]] = []
    rec_of: dict[int, dict[str, Any]] = {}  # id() of a candidate made from a record -> record
    olds: list[dict[str, Any]] = []  # the crash moments there were
    for rec in cands:
        i = rec.get("source_index")
        if i not in parts or is_pool(rec):
            rest.append(rec)
        elif is_nominated(rec):  # analyze builds crash moments before events nominate
            for k in CRASH_MARKS:
                rec.pop(k, None)
            rest.append(rec)
        elif rec.get("kind") == CRASH_KIND:
            olds.append(rec)
        else:
            c = Candidate(parts[i].source, rec["start"], rec["end"], rec["start"], 0.0, {})
            parts[i].candidates.append(c)
            rec_of[id(c)] = rec
    add_crash_candidates(list(parts.values()), style, load_crashes(project))
    kept: list[dict[str, Any]] = []
    built: list[dict[str, Any]] = []  # the crash moments made now
    for i, part in parts.items():
        for c in part.candidates:
            rec = rec_of.get(id(c))
            if rec is not None:  # a candidate of the analysis: with its crash mark, or none
                for k in CRASH_MARKS:
                    rec.pop(k, None)
                if c.event:
                    rec["crash"] = c.event
                    rec["crash_confidence"] = round(float(c.event["confidence"]), 3)
                    rec["crash_forced"] = c.must_include
                kept.append(rec)
                continue
            built.append(_candidate_record(c, i, style))
    was = _crash_keys(olds)
    fresh: list[dict[str, Any]] = []  # crash moments with no earlier record
    for key, new in _crash_keys(built).items():
        old = was.get(key)
        if old is None:
            fresh.append(new)
            continue
        # the same crash: as it was (a span menu time trimmed included) if its marks are the
        # same, else the new moment with its id
        same = all(old.get(k) == new.get(k) for k in (*CRASH_MARKS, "must_include"))
        kept.append(old if same else ({"id": old["id"]} if "id" in old else {}) | new)
    ev = cache.load_json(project / "events.json")
    if fresh and ev:  # menu time drops or trims a new one as analyze does, so it takes no id
        menu_hl = {
            "style": hl.get("style"),
            "style_base": hl.get("style_base"),
            "settings": hl.get("settings"),
            "candidates": fresh,
        }
        drop_menu_time(menu_hl, ev)
        fresh = menu_hl["candidates"]
    pool = [c for c in rest if is_pool(c)]
    kept = sorted(
        [c for c in rest if not is_pool(c)] + kept + fresh,
        key=lambda c: (not c.get("must_include"), *scorer.rank_key(c), *scorer.tie_key(c)),
    )
    hl["candidates"] = kept + pool
    number_candidates(hl)
    if _canonical(hl) == _canonical(before):  # the same records, maybe in another order
        return False
    cache.save_json(path, hl)
    logger.info("Crash moments rebuilt from crashes.json in %s", path.name)
    return True


def band_k(suggested_top: int) -> int:
    """How many moments the vision band takes below the default edit's cut (#909):
    :data:`VISION_BAND_SHARE` of the edit's ``suggested_top`` moments, rounded up, within
    :data:`VISION_BAND_MIN` and :data:`VISION_BAND_MAX`."""
    k = math.ceil(round(VISION_BAND_SHARE * suggested_top, 9))  # no float error can add one
    return min(VISION_BAND_MAX, max(VISION_BAND_MIN, k))


def vision_band(
    ranked: list[dict[str, Any]], picks: list[dict[str, Any]], max_n: int | None
) -> tuple[list[dict[str, Any]], int, int]:
    """The moments the vision model rates (#909): the default edit and ``k`` below its cut.

    ``ranked``: the candidates that are neither ``must_include`` nor visual pool moments, in
    :func:`cutpoint.scorer.edit_rank_key` order. ``picks``: the default edit
    (:func:`cutpoint.default_fit.fit_target` before vision). The cut is 1 + the index in
    ``ranked`` of the last pick that is not ``must_include`` (the picks need not be the first
    ones; 0 without one). The band is every pick in ``ranked`` (a moment above the cut that the
    edit skipped for its length or because it overlaps a pick is not one) plus ``ranked[cut : cut + k]``, ``k`` =
    :func:`band_k` of the number of picks, clipped at the end. Closest to the cut first: the
    last pick, the first moment below the cut, the pick before the last, the second below, and
    so on, the rest of one side once the other runs out; then cut to ``max_n``
    (``--vision-top``; None: all). Returns ``(band, cut, k)``."""
    where = {id(c): i for i, c in enumerate(ranked)}
    at = sorted(
        {where[id(c)] for c in picks if not c.get("must_include") and id(c) in where},
        reverse=True,
    )
    cut = at[0] + 1 if at else 0
    k = band_k(len(picks))
    above = [ranked[i] for i in at]
    below = ranked[cut : cut + k]
    order: list[dict[str, Any]] = []
    for i in range(max(len(above), len(below))):
        order.extend(side[i] for side in (above, below) if i < len(side))
    return (order if max_n is None else order[:max_n]), cut, k


def _find_input(target: Path, exclude: Sequence[str], recursive: bool) -> list[Path]:
    """``find_videos`` for one input, read through ``source_remap`` when it is missing.

    A missing file or folder whose ``source_remap`` target exists is listed there;
    each video found is returned under its original spelling (below ``target``), and
    ``exclude`` is matched against that spelling, as for an input still in place.
    A video in ``source_remap_exclude`` is skipped with a warning: it would be read
    at its missing original.

    Raises:
        FileNotFoundError: As ``find_videos``, when neither path can be listed.
    """
    # os.path.exists: any OSError counts as missing; find_videos then reports it.
    io = target if os.path.exists(target) else remap_source(target)  # noqa: PTH110
    if io == target:
        return find_videos(target, exclude=exclude, recursive=recursive)
    logger.info("source_remap: %s -> %s", target, io)
    # UNC form for a disconnected mapped drive, as find_videos records a reachable one.
    base, orig = io.resolve(), Path(unc_path(str(target.resolve())))
    found: list[Path] = []
    for read in find_videos(io, recursive=recursive):
        # A symlink resolved outside the folder keeps the path that was read.
        p = orig / read.relative_to(base) if read.is_relative_to(base) else read
        if matches_glob(p, exclude):
            continue
        if p != read and remap_source(p) == p:  # source_remap_exclude: not read via the remap
            logger.warning(
                "Skipping %s: in source_remap_exclude and missing at its original path", p
            )
            continue
        found.append(p)
    return found


def _find_inputs(
    target: Path | Sequence[Path], exclude: Sequence[str], recursive: bool
) -> list[Path]:
    """Videos of every input (see :func:`_find_input`), de-duplicated, in input order."""
    targets = [target] if isinstance(target, (str, Path)) else target
    found = [p for t in targets for p in _find_input(Path(t), exclude, recursive)]
    return list(dict.fromkeys(found))


def find_sources(
    target: Path | Sequence[Path], exclude: Sequence[str], recursive: bool
) -> list[Path]:
    """The sources ``analyze`` reads for ``target`` (:func:`_find_inputs`), de-duplicated by
    the file actually read (``source_remap``; the first spelling given is the one recorded).

    Raises:
        FileNotFoundError: If no video is found.
    """
    found = _find_inputs(target, exclude, recursive)
    # Projects record the original paths; files are read through config source_remap.
    by_read: dict[Path, Path] = {}
    for s in found:
        by_read.setdefault(remap_source(s), s)
    sources = list(by_read.values())
    for orig in sources:
        if (src := remap_source(orig)) != orig:
            logger.info("source_remap: %s -> %s", orig, src)
    if not sources:
        raise FileNotFoundError(
            f"No video files found in {target} (exclude={list(exclude)}, recursive={recursive})"
        )
    logger.info("%d source(s): %s", len(sources), ", ".join(s.name for s in sources))
    return sources


# --------------------------------------------------------------------------- analysis steps
GAME_STYLES = ("gaming", "funny")  # keyframes: HUD OCR (with region discovery)
ACTION_STYLES = ("action", "sport", ACTION_CAM)  # keyframes: CLIP; crash detection
# Skip reasons of plan_layers (the PROGRESS ``detail``) and, for keyframes, why
# crashes.json has no visual layer (its ``no_visual_layer`` warning).
SKIP_OPTION = "left out by --skip or analysis.skip"
SKIP_NO_EXTRA = "events extra not installed"
SKIP_NO_CUDA = "no CUDA GPU (CLIP on the CPU: about an hour per hour)"
NO_VISUAL_REASON = {
    SKIP_OPTION: "keyframes left out by --skip or analysis.skip",
    SKIP_NO_EXTRA: "the events extra is not installed",
    SKIP_NO_CUDA: "no CUDA GPU for the keyframes step",
}
# How to avoid HUD discovery during analyze (no --game or --no-discover here), after "Put a
# preset in <presets dir>" (run_events' discover_hint)
DISCOVER_HINT = "or pass --skip keyframes to leave out the whole keyframes step"
LAYER_STEP = {
    "audio": "reactions",
    "voice": "reactions",
    "visual": "keyframes",
    "ocr": "keyframes",
    "crash": "crashes",
    "dialogue": "dialogue",  # the dialogue reader, run by run_events before it writes
    "write": "ranking",
}


@dataclass
class LayerPlan:
    """The event layers ``analyze`` runs per step, and why the other steps are left out.

    Attributes:
        layers: ``{step: layers}`` of the steps that run (:data:`LAYER_STEP` order).
        skipped: ``{step: reason}`` of the steps left out.
        game: Game footage: cached HUD OCR events count even when OCR does not run.
    """

    layers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    game: bool = False

    def all_layers(self) -> tuple[str, ...]:
        """Every planned layer, in the order :func:`run_events` runs them."""
        return tuple(x for x in LAYER_STEP if any(x in v for v in self.layers.values()))


def has_telemetry(src: Path, timeout: float = 60) -> bool:
    """Whether ``src`` carries camera telemetry (a GoPro GPMF track or an Insta360 X3 IMU
    trailer); only the container's stream list is read, within ``timeout`` seconds."""
    from cutpoint.insta360 import imu_source  # noqa: PLC0415

    try:
        if imu_source(src) is not None:
            return True
        info = ffprobe_json(remap_source(src), entries="stream=codec_tag_string", timeout=timeout)
    except (OSError, FFmpegError):
        return False
    return any(s.get("codec_tag_string") == "gpmd" for s in info.get("streams") or [])


# The --json warning of action-cam sources without camera telemetry (#939).
NO_TELEMETRY_WARNING = "action_cam_no_telemetry"


def _no_telemetry_fallback(style: Style, sources: Sequence[Path]) -> tuple[Style, dict[str, str]]:
    """The ``action`` style an action-cam ``style`` falls back to when no source has camera
    telemetry (with the ``--must-include-crash`` of ``style``), and the
    ``settings.style_fallback`` record; logs a WARNING and the ``--json`` warning
    :data:`NO_TELEMETRY_WARNING` naming every source."""
    from cutpoint.styles import get_style  # noqa: PLC0415

    used = replace(get_style("action"), must_include_crash=style.must_include_crash)
    named = "action-cam" if style.name == ACTION_CAM else f"{style.name} (an action-cam style)"
    message = (
        f"No camera telemetry in any source; {named} needs GoPro or Insta360 X3 telemetry, "
        f"so the {used.name} style was used. Give --style {used.name} (or auto) to choose it"
    )
    logger.warning("%s", message)
    jsonout.warn(NO_TELEMETRY_WARNING, message, sources)
    return used, {"from": style.name, "reason": "no telemetry"}


def _warn_missing_telemetry(missing: Sequence[Path]) -> None:
    """Warn (WARNING and the ``--json`` warning :data:`NO_TELEMETRY_WARNING`) of the sources
    of an action-cam analysis without camera telemetry, or whose telemetry could not be read:
    their candidates come from sound, speech and the visual features alone, scored over those
    so they rank with the others (#950). Given once per ``--json`` command: its re-rank
    (``360 analyze``, ``edit --events``) adds no second one."""
    if not missing or jsonout.has_warning(NO_TELEMETRY_WARNING, missing):
        return
    message = (
        f"{len(missing)} source(s) have no camera telemetry (or it could not be read); their "
        "moments come from sound, speech and visual features, scored over those so they "
        "rank with the moments of the sources that have it"
    )
    logger.warning("%s", message)
    jsonout.warn(NO_TELEMETRY_WARNING, message, missing)


def _cuda() -> bool:
    """Whether torch sees a CUDA GPU; False when torch itself fails to load (a broken
    install, a missing DLL), so the analysis goes on without the GPU-only steps."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        logger.debug("torch CUDA check failed", exc_info=True)
        return False


def layer_category(style: Style) -> str:
    """The footage category (:data:`cutpoint.category.CATEGORIES`) whose layers an explicit
    ``style`` runs, from today's style sets: :data:`GAME_STYLES` gameplay, the action-cam
    kind action cam, the other :data:`ACTION_STYLES` action, any other style talk."""
    if style.kind == ACTION_CAM:
        return category.ACTION_CAM
    if style.base in GAME_STYLES:
        return category.GAMEPLAY
    if style.base in ACTION_STYLES:
        return category.ACTION
    return category.TALK


def plan_layers(
    style: Style,
    sources: Sequence[Path],
    skip: Collection[str],
    cat: str | None = None,
    visual_index: bool = False,
) -> LayerPlan:
    """Which event layers the reactions, keyframes and crashes steps run.

    ``cat``: the footage category (:mod:`cutpoint.category`; None: :func:`layer_category`
    of ``style``). Reactions: the PANNs audio events (plus the voice layer for action-cam
    footage, whose crash audio score reads it). Keyframes: HUD OCR for gameplay, the CLIP
    visual layer for action-cam, action and general footage (only with a CUDA GPU); ``visual_index``
    (``--visual-index``) adds the CLIP visual layer to the keyframes of gameplay (before the
    HUD text) and talk too, for the assistant's visual search only. Crashes:
    action-cam, action and general footage, or any source with telemetry. A step whose events extra
    or model files are missing is left out with one INFO line naming the fix; nothing is
    downloaded.
    """
    cat = cat or layer_category(style)
    plan = LayerPlan(game=cat == category.GAMEPLAY)
    action = cat in (category.ACTION_CAM, category.ACTION, category.GENERAL)
    for step in SKIPPABLE:
        if step in skip:
            plan.skipped[step] = SKIP_OPTION
    want = plan.layers
    if "crashes" not in plan.skipped:
        if action or any(has_telemetry(s) for s in sources):
            want["crashes"] = ("crash",)
        else:
            plan.skipped["crashes"] = "not action-cam footage"
    if "reactions" not in plan.skipped:
        want["reactions"] = ("audio", "voice") if "crashes" in want else ("audio",)
    if "keyframes" not in plan.skipped:
        if plan.game:
            want["keyframes"] = ("ocr", "visual") if visual_index else ("ocr",)
        elif action or visual_index:
            want["keyframes"] = ("visual",)
        else:
            plan.skipped["keyframes"] = f"no keyframe layer for the {style.name} style"
    if not want:
        return plan
    missing = [m for m in EVENTS_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        from cutpoint.extras import LITE, install_hint  # noqa: PLC0415

        names = [LABELS[s].lower() for s in want]
        logger.info(
            "The %s %s %s (the events extra is not installed) and skipped; to add it: %s",
            " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names),
            "step is" if len(names) == 1 else "steps are",
            LITE,
            install_hint("events"),
        )
        for step in want:
            plan.skipped[step] = SKIP_NO_EXTRA
        want.clear()
        return plan
    if "visual" in want.get("keyframes", ()) and not _cuda():
        rest = tuple(x for x in want["keyframes"] if x != "visual")  # gameplay: the HUD text stays
        logger.info(
            "No CUDA GPU: %s skipped (CLIP on the CPU would take about an hour per hour of "
            "footage)",
            "keyframes" if not rest else "the visual index",
        )
        if rest:
            want["keyframes"] = rest
        else:
            plan.skipped["keyframes"] = SKIP_NO_CUDA
            del want["keyframes"]
    components: dict[str, None] = {}
    for step, layers in list(want.items()):
        files = event_paths.missing_files(layers)
        if not files:
            continue
        components.update(dict.fromkeys(comp for comp, _ in files))
        # reactions without the voice layer, keyframes of gameplay without the visual index
        rest = tuple(x for x in layers if x not in ("voice", *(("visual",) if plan.game else ())))
        if rest != layers and rest and not event_paths.missing_files(rest):
            want[step] = rest
            continue
        need = " ".join(dict.fromkeys(comp for comp, _ in files))
        plan.skipped[step] = f"model files missing (cutpoint setup --only {need})"
        del want[step]
    if components:
        logger.info(
            "Model files are missing for some analysis steps (nothing is downloaded now); "
            "to add them: cutpoint setup --only %s",
            " ".join(sorted(components, key=COMPONENTS.index)),
        )
    return plan


# An error text of a failed allocation: torch ("CUDA out of memory"), onnxruntime
# ("Failed to allocate memory"), OpenCV ("Insufficient memory"), C++ (std::bad_alloc).
OOM_RE = re.compile(
    r"out of memory|insufficient memory|(?:unable|failed) to allocate|bad[ _]alloc", re.I
)
REASON_CHARS = 120  # the reason of a failed OCR layer is cut to this


def _ocr_failed(err: BaseException) -> tuple[str, list[str]]:
    """The WARNING (and ``step_failed`` message) of an OCR layer that failed with ``err``,
    naming what it costs, and the paths of the sources it did not read (none when the
    failure is not tagged with them, :func:`~cutpoint.events.pipeline.ocr_failure`). The
    out-of-memory hint names ``CUTPOINT_ANALYZE_OVERLAP`` only when the failing read ran in
    the background while Whisper ran (:func:`_ocr_prefetch`)."""
    from cutpoint.events.pipeline import ocr_failure  # noqa: PLC0415

    lines = str(err).strip().splitlines()
    oom = isinstance(err, MemoryError) or bool(OOM_RE.search(str(err)))
    if oom:
        reason = "out of memory"
    else:
        reason = lines[0].strip() if lines else type(err).__name__
        if len(reason) > REASON_CHARS:
            reason = reason[: REASON_CHARS - 3].rstrip() + "..."
    where = ocr_failure(err)
    paths = where.not_read if where else []
    effect = "menu filtering and HUD events (kill feed, banners, notices) skipped for"
    if where is None:  # setup or bookkeeping outside the reads: which sources is not known
        message = f"OCR failed ({reason}): {effect} the sources not read yet"
    elif where.discovery:
        games = " for some games" if where.games > 1 else ""
        message = (
            f"OCR failed in HUD region discovery ({reason}): no HUD preset installed{games}, "
            f"{effect} every source"
        )
    elif not paths:
        message = f"OCR failed ({reason}): {effect} every source"
    else:
        later = len(paths) - 1
        rest = f" and {later} later source{'s' if later != 1 else ''}" if later else ""
        message = f"OCR failed ({reason}): {effect} {Path(paths[0]).name}{rest}"
    if oom:
        hint = ", set CUTPOINT_ANALYZE_OVERLAP=0" if where and where.background else ""
        message += f"; close other programs{hint} or pass --skip keyframes"
    return message, paths


class _StepHook:
    """``run_events``'s ``stage`` for :func:`analyze`: each layer reports as its step; a
    layer that fails fails its step (an ERROR line and a ``step_failed`` warning; for the
    OCR layer a WARNING naming the sources without menu filtering and HUD events, see
    :func:`_ocr_failed`), and the next steps still run. A layer in ``optional`` that fails
    only gets a WARNING and a ``step_failed`` warning: its step is not failed."""

    def __init__(self, steps: progress.Run, optional: Collection[str] = ()) -> None:
        self.steps = steps
        self.optional = optional
        self.failed: set[str] = set()

    @contextlib.contextmanager
    def __call__(self, layer: str) -> Iterator[None]:
        step = LAYER_STEP[layer]
        if step not in self.failed:
            self.steps.start(step)
        if layer == "write":  # events.json: an error here stops the run
            yield
            return
        try:
            yield
        except LowMemoryError:  # the memory guard stops the run (resumable), not the layer
            raise
        except Exception as err:
            logger.debug("%s layer failed", layer, exc_info=True)
            files: list[str] = []
            if layer == "ocr":  # analyze goes on, ranked without the HUD: a WARNING
                message, files = _ocr_failed(err)
                logger.warning("%s", message)
            elif layer in self.optional:
                message = f"Visual index failed: {err}; the HUD text still runs"
                logger.warning("%s", message)
            else:
                message = f"{LABELS[step]} failed: {err}; continuing without it"
                logger.error("%s", message)
            jsonout.warn("step_failed", message, files)
            if layer in self.optional:
                return
            if self.steps.current == step:
                self.steps.fail(message)
            self.failed.add(step)


def _ocr_options(project: Path, *, log: bool = True) -> tuple[tuple[str, ...], str]:
    """Player tags and HUD game for the OCR layer. Tags: ``player_tags`` in config.json,
    or when it has none those of the last ``cutpoint events`` run (its ``--player``, kept
    in events.json). Game: that run's ``--game`` when it was not ``auto``. ``log`` False:
    no INFO line about what was reused (the OCR prefetch asks before its turn)."""
    from cutpoint.config import player_tags  # noqa: PLC0415

    tags, game = player_tags(), "auto"
    doc = cache.load_json(project / "events.json")
    settings = doc.get("settings") if isinstance(doc, dict) else None
    prev = settings.get("ocr") if isinstance(settings, dict) else None
    if not isinstance(prev, dict):
        return tags, game
    reused = []
    old = prev.get("tags")
    if not tags and isinstance(old, list) and old and all(isinstance(t, str) for t in old):
        tags = tuple(old)
        reused.append(f"{len(tags)} player tag{'s' if len(tags) != 1 else ''}")
    old_game = prev.get("game")
    if isinstance(old_game, str) and old_game and old_game != game:
        game = old_game
        reused.append(f"game {game}")
    if reused and log:
        logger.info("Reusing %s from the last `cutpoint events` run", " and ".join(reused))
    return tags, game


def _ocr_prefetch(
    style: Style,
    sources: Sequence[Path],
    skip: Collection[str],
    project: Path,
    layers: bool,
    cat: str | None = None,
    batch_size: int = 0,
    det_skip: bool = False,
) -> contextlib.AbstractContextManager[Any]:
    """The background OCR reader of :mod:`cutpoint.events.ocr_prefetch` when the keyframes
    step will run HUD OCR (the conditions of :func:`plan_layers` with the category ``cat``,
    checked without its log lines, which keep their place) and ``CUTPOINT_ANALYZE_OVERLAP``
    is not ``0``; else a context giving None. ``batch_size``: ``--batch-size`` (its engine
    is made again every this many sources); ``det_skip``: ``--ocr-det-skip``."""
    gameplay = (cat or layer_category(style)) == category.GAMEPLAY
    if not layers or not gameplay or "keyframes" in skip:
        return contextlib.nullcontext()
    if any(importlib.util.find_spec(m) is None for m in EVENTS_MODULES):
        return contextlib.nullcontext()
    if event_paths.missing_files(("ocr",)):
        return contextlib.nullcontext()
    from cutpoint.events import ocr_prefetch  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415

    if not ocr_prefetch.enabled():
        return contextlib.nullcontext()
    tags, game = _ocr_options(project, log=False)
    params = OcrParams(game=game, player_tags=tags, det_skip=det_skip)
    return ocr_prefetch.OcrPrefetch(project, sources, params, batch_size=batch_size)


def _scene_plan(src: Path, duration: float) -> Any:
    """The :class:`~cutpoint.events.ocr_prefetch.ScenePlan` of a whole source whose scene
    proxy may be made in the background: up to :data:`LONG_FILE_S` (longer files decode
    keyframes only), its scene cuts not cached, one video stream (no cover art) and a
    container that seeks exactly (:func:`_seeks_exactly`: an MPEG-TS pass places its first
    frame by a decode of its own, :func:`_first_shift`); else None (an INFO line names a
    failed stream or container gate; called in a shared-decode mode only)."""
    from cutpoint.events.ocr_prefetch import ScenePlan  # noqa: PLC0415

    cdir = cache.source_dir(src)
    if duration > LONG_FILE_S or cache.load_json(cdir / SCENES_FILE) is not None:
        return None
    io = remap_source(src)
    if not single_video_stream(io):
        logger.info(
            "%s: shared decode off: not exactly one video stream (cover art or several)",
            src.name,
        )
        return None
    if not _seeks_exactly(io):
        logger.info(
            "%s: shared decode off: MPEG-TS or another container without exact seeks", src.name
        )
        return None
    _pre, _post, vf, enc = _proxy_args(duration)
    return ScenePlan(io, cdir, duration, vf, tuple(enc))


def _dialogue_backend(
    style: Style,
    skip: Collection[str],
    steps: progress.Run,
    transcripts: dict[str, list[dict[str, Any]]],
    ollama_url: str | None,
    ollama_model: str | None,
) -> Any:
    """The text model that reads the transcripts for the ``dialogue`` step
    (:func:`cutpoint.events.dialogue.choose_backend`, made once per run), or None with the step
    planned as skipped and the reason logged. Open-list styles only."""
    from cutpoint.events import dialogue  # noqa: PLC0415

    if "dialogue" in skip:  # planned by plan_layers
        return None
    if not style.open_list:
        steps.skip("dialogue", f"for open-list styles only, not the {style.name} style")
        return None
    if not any(s.get("text") for segs in transcripts.values() for s in segs):
        steps.skip("dialogue", "no speech to read")
        return None
    choice = dialogue.choose_backend(ollama_url, ollama_model)
    if choice.backend is None:
        logger.info("Dialogue step skipped: %s", choice.reason)
        steps.skip("dialogue", choice.reason)
    return choice.backend


def _dialogue_reader(
    backend: Any,
    transcripts: dict[str, list[dict[str, Any]]],
    steps: progress.Run,
    timings: dict[str, float],
    ran: list[bool],
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """The dialogue step as a function of the audio events (events.json entries with source,
    ``start``, ``label`` and ``confidence``): reads the ``transcripts`` with ``backend`` and
    returns the dialogue events. Records ``timings["dialogue_s"]`` and appends to ``ran`` when
    called."""
    from cutpoint.events import dialogue  # noqa: PLC0415

    def read(audio: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ran.append(True)
        t0 = time.perf_counter()
        try:
            found = dialogue.read_all(
                transcripts, audio, backend, lambda pct, name: steps.update(pct, name)
            )
        except LowMemoryError:  # an optional step never aborts the analysis
            message = f"Dialogue skipped: not enough free memory for {backend.model}"
            logger.warning("%s", message)
            jsonout.warn("step_failed", message)
            found = []
        timings["dialogue_s"] = round(time.perf_counter() - t0, 2)
        return found

    return read


def _audio_events(layer1: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """The audio events of the layer-1 events per source, as events.json has them."""
    return [
        {
            "kind": "audio",
            "source": src,
            "start": e["t"],
            "label": e.get("label"),
            "confidence": e.get("confidence"),
        }
        for src, evs in layer1.items()
        for e in evs
        if e.get("kind") == "audio"
    ]


def _dialogue_alone(
    project: Path,
    hl: dict[str, Any],
    transcript: str,
    steps: progress.Run,
    reader: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    sources: list[str],
) -> None:
    """The ``dialogue`` step when the event layers did not run it: its events replace the
    earlier ones in events.json (made if there is none) and events.txt. The audio events it
    ranks by are those events.json already has, if any. An error fails the step only."""
    from cutpoint.events import report  # noqa: PLC0415
    from cutpoint.events.common import pending_highlights  # noqa: PLC0415

    with _StepHook(steps)("dialogue"):
        doc = cache.load_json(project / "events.json")
        old = doc.get("events") if isinstance(doc, dict) else None
        audio = [e for e in old or [] if isinstance(e, dict) and e.get("kind") == "audio"]
        found = reader(audio)
        with pending_highlights(project, hl, transcript):
            report.replace_dialogue(project, found, list(dict.fromkeys(sources)))


def _run_layers(
    project: Path,
    plan: LayerPlan,
    hl: dict[str, Any],
    steps: progress.Run,
    llm: bool,
    prefetch: OcrPrefetch | None = None,
    transcript: str | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    batch_size: int = 0,
    dialogue: Callable[[dict[str, list[dict[str, Any]]]], list[dict[str, Any]]] | None = None,
    det_skip: bool = False,
) -> bool:
    """Run the planned event layers on the analysis ``hl`` (not written yet) and write
    events.json, as ``cutpoint events`` would (held back with the other
    :data:`HELD_OUTPUTS` until the analysis completes); False if they stopped before it.
    ``prefetch``: the background OCR reader (:func:`_ocr_prefetch`) or None; ``transcript``:
    the transcript.txt text, not written yet either (None: read from the project);
    ``ollama_url``, ``ollama_model``: the VLM layer's Ollama (``analyze --ollama-url``, #660;
    None: from config.json and the environment); ``batch_size``: ``--batch-size``
    (:func:`~cutpoint.events.pipeline.run_events`); ``dialogue``: the dialogue reader it runs as
    the ``dialogue`` step before it writes events.json (:func:`_dialogue_reader`); ``det_skip``:
    ``--ocr-det-skip`` (``OcrParams.det_skip``)."""
    from cutpoint.crash import no_visual_reason  # noqa: PLC0415
    from cutpoint.events.audio_events import AudioParams  # noqa: PLC0415
    from cutpoint.events.common import pending_highlights  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415
    from cutpoint.events.pipeline import run_events  # noqa: PLC0415
    from cutpoint.events.vlm_windows import VlmParams  # noqa: PLC0415

    tags, game = _ocr_options(project)
    why = plan.skipped.get("keyframes")
    try:
        with (
            pending_highlights(project, hl, transcript),
            no_visual_reason(NO_VISUAL_REASON.get(why or "")),
            prefetch or contextlib.nullcontext(),  # stopped once the layers are done
        ):
            # The options of a plain `cutpoint events`, so it reuses these caches.
            run_events(
                project,
                layers=plan.all_layers(),
                audio=AudioParams(),
                ocr=OcrParams(game=game, player_tags=tags, det_skip=det_skip),
                vlm=VlmParams(max_windows=None),
                stage=_StepHook(steps, {"visual"} if plan.game else set()),
                cached_ocr=plan.game,
                llm=llm,
                prefetch=prefetch,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                discover_hint=DISCOVER_HINT,
                batch_size=batch_size,
                dialogue=dialogue,
            )
    except Exception as err:
        if steps.current == "ranking" or isinstance(err, LowMemoryError):
            raise
        message = f"Analysis steps failed: {err}; continuing without them"
        logger.debug("event layers failed", exc_info=True)
        logger.error("%s", message)
        jsonout.warn("step_failed", message)
        if steps.current is None:  # stopped before its first layer: that step failed
            first = next((s for s in SKIPPABLE if s in plan.layers and s not in steps.state), None)
            if first is not None:
                steps.start(first)
        if steps.current is not None:
            steps.fail(message)
        for step in plan.layers:
            steps.skip(step, "not run: an earlier step failed")
        return False
    return True


def free_models() -> None:
    """Free a closed model's memory before the next step loads one (and torch's cached
    CUDA memory when torch is loaded)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _features_by_source(results: list[SourceResult]) -> dict[str, list[str]]:
    """``settings.features_by_source``: per source path (as in the candidates' ``source``), the
    sorted features its scores were divided by. A source not picked from gets the install's set."""
    out: dict[str, list[str]] = {}
    for r in results:
        out[str(r.source)] = sorted(features.available() if r.avail is None else r.avail)
    return out


def _candidate_record(c: Candidate, source_index: int, style: Style) -> dict[str, Any]:
    """A highlights.json candidate without its ``id`` (given by the final rank)."""
    return {
        "source": str(c.source),
        "source_index": source_index,
        "start": c.start,
        "end": c.end,
        "duration": round(c.end - c.start, 3),
        "score": c.score,
        "signals": {
            **c.signals,
            "keywords": c.keywords,
            "vision": None,  # set by _rate_vision
        },
        "label": _label(c, style),
        **({"kind": c.kind} if c.kind else {}),
        **({"must_include": True} if c.must_include else {}),
        **(
            {
                "crash": c.event,
                "crash_confidence": round(float(c.event["confidence"]), 3),
                "crash_forced": c.must_include,
            }
            if c.event
            else {}
        ),
        "transcript": c.transcript,
        **({"transcript_unsure": c.transcript_unsure} if c.transcript_unsure else {}),
    }


def _visual_pool(
    project: Path, results: Sequence[CrashPart], style: Style, hl: dict[str, Any]
) -> list[dict[str, Any]]:
    """The visual pool candidates of a run (:mod:`cutpoint.visual_pool`), ids ``v001``...,
    recorded in ``hl``'s ``settings.visual_pool``. A failure logs a WARNING and adds none."""
    from cutpoint import visual_pool  # noqa: PLC0415
    from cutpoint.crash import load_crashes  # noqa: PLC0415

    parts = [
        visual_pool.Part(r.source, i, r.offset, r.duration, r.signals.n)
        for i, r in enumerate(results)
        if r.signals is not None
    ]
    # spoken spans in source time, from the words already transcribed (a tie-break only)
    speech = {
        i: [(x["start"] + r.offset, x["end"] + r.offset) for x in r.signals.words]
        for i, r in enumerate(results)
        if r.signals is not None
    }
    try:
        found = visual_pool.build(
            project, parts, style, hl["candidates"], load_crashes(project), speech
        )
    except Exception as err:  # optional: never fails the analysis
        logger.debug("visual pool failed", exc_info=True)
        logger.warning("Visual candidates skipped: %s", err)
        found = []
    out = []
    for i, w in enumerate(found, 1):
        r = results[w.index]
        words = r.signals.words if r.signals is not None else []
        said = [x for x in words if w.start - r.offset <= x["start"] <= w.end - r.offset]
        transcript, unsure = _transcript_snippet(said)
        out.append(
            {
                "id": f"v{i:03d}",
                "source": str(w.source),
                "source_index": w.index,
                "start": w.start,
                "end": w.end,
                "duration": round(w.end - w.start, 3),
                "score": round(10 * w.peak, 1),
                "signals": w.signals,
                "label": visual_pool.LABELS[w.kind],
                "kind": w.kind,
                "pool": visual_pool.POOL,
                "transcript": transcript,
                **({"transcript_unsure": unsure} if unsure else {}),
            }
        )
    kinds = {k: sum(c["kind"] == k for c in out) for k in visual_pool.KINDS}
    hl["settings"]["visual_pool"] = {"count": len(out), "kinds": kinds}
    if out:
        logger.info(
            "Visual candidates: %d (%s)",
            len(out),
            ", ".join(f"{n} {k}" for k, n in kinds.items() if n),
        )
    return out


def vision_budget(
    share: float, pre_vision_s: float, earlier_total_s: float | None
) -> tuple[float, str]:
    """The vision step's time budget (#909): ``share`` (``vision_time_share``) of the analysis
    before it. A re-rank measures against the first analysis (``earlier_total_s``, its
    ``timings.total_s``), as its own run before vision is mostly cached. Returns
    ``(budget_s, the name of the base)``."""
    if earlier_total_s is not None:
        return share * earlier_total_s, "earlier_total_s"
    return share * pre_vision_s, "pre_vision_s"


def cap_reached(now: float, deadline: float, calls: int) -> bool:
    """Whether the vision time cap stops further model calls: at least one call was made and
    the clock ``now`` is at or past the ``deadline``."""
    return calls >= 1 and now >= deadline


@dataclass
class VisionOutcome:
    """What :func:`_rate_vision` did: ``used`` any rating applied; ``failed`` moments without a
    rating (cache misses of a ``cache_only`` rater included); ``capped`` the time cap stopped
    the model calls and ``skipped_by_cap`` moments were left unrated for it; ``centres`` per
    prompt kind (:func:`rating_centres`); ``by_kind`` the moments rated per prompt kind."""

    used: bool = False
    failed: int = 0
    capped: bool = False
    skipped_by_cap: int = 0
    centres: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, int] = field(default_factory=dict)


def _rate_vision(
    order: list[dict[str, Any]],
    rater: VisionRater,
    style: Style,
    activity: Prompts | None = None,
    batch_size: int = 0,
    *,
    deadline: float,
    clock: Callable[[], float] = time.perf_counter,
) -> VisionOutcome:
    """Rate the candidates of ``order`` (the vision band, closest to the cut first,
    :func:`vision_band`) with the vision model and apply the ratings to their analysis score
    (``score_pre_events`` once events rescored them).

    The model is asked in ``order`` until the time cap: once a call was made and ``clock()``
    reached ``deadline`` (:func:`cap_reached`), a moment without a cached rating is skipped; a
    cached one is still taken (it costs nothing). A call under way when the deadline passes
    finishes, so the step can overrun its budget by one call.

    Then each prompt kind's ratings (``"speech"`` when the moment has speech to send with its
    frames, :func:`~cutpoint.vision.speech_excerpt`, else ``"frames"``) are centred on their
    mean (:func:`rating_centres` with the prior of the style's :func:`prior_group`,
    :func:`centred_rating`) and blended in with
    :func:`vision_blend` and the style's ``vision_weight``. ``signals.vision`` keeps the
    model's rating as given; its label goes to ``vision_label``, never to ``label`` (#913).
    Each rated candidate keeps ``vision_rated`` (the span rated, the score before and after the
    blend, the prompt kind and its centre) for :func:`sync_vision`.
    ``activity`` words the action-cam prompt (the project's activity
    prompt set). ``batch_size``: ``--batch-size`` (#713), the model is unloaded after the
    candidates of every this many sources (:func:`_vision_batches`); the ratings are the same,
    but once the time cap is reached, which moments near the cut get rated can differ."""
    out = VisionOutcome()

    def cached(c: dict[str, Any]) -> bool:
        return (
            rater.cached(
                remap_source(Path(c["source"])),
                c["start"],
                c["end"],
                style.base,
                activity=activity,
                transcript=c.get("transcript"),
                laughter=heard_laughter(c),
            )
            is not None
        )

    if not rater.cache_only:  # a resumed run: the ratings made before are cached
        progress.continuing("vision", sum(cached(c) for c in order), len(order))
    got: list[tuple[dict[str, Any], float, str | None]] = []
    done = calls = 0
    try:
        for k, part in enumerate(_vision_batches(order, batch_size)):
            if k:  # a batch boundary: memory back until the next request loads the model
                rater.unload()
                progress.continuing("vision", done, len(order))
            for c in part:
                progress.update(progress.fraction(done, len(order)))
                done += 1
                if cap_reached(clock(), deadline, calls) and not cached(c):
                    out.skipped_by_cap += 1
                    continue
                hits = rater.hits
                res = rater.rate(
                    remap_source(Path(c["source"])),
                    c["start"],
                    c["end"],
                    style.base,
                    activity=activity,
                    transcript=c.get("transcript"),
                    laughter=heard_laughter(c),
                )
                if rater.hits == hits and not rater.cache_only:
                    calls += 1
                if res is None:
                    out.failed += 1
                    continue
                got.append((c, res.score, res.label))
                logger.info(
                    "vision %.1f '%s' for %.1f-%.1f", res.score, res.label, c["start"], c["end"]
                )
    finally:  # free the VRAM now, not after Ollama's default 5 min keep_alive
        rater.unload()
    out.capped = out.skipped_by_cap > 0
    logger.info("Vision: %d ratings from the cache, %d asked the model", rater.hits, rater.asked)
    # each prompt kind centred on this run's ratings of it
    kinds = [
        "speech" if speech_excerpt(c.get("transcript"), style.base) else "frames" for c, _, _ in got
    ]
    out.centres = rating_centres(
        ((kind, r) for kind, (_, r, _) in zip(kinds, got, strict=True)), prior_group(style.base)
    )
    for kind, (c, score, label) in zip(kinds, got, strict=True):
        out.used = True
        out.by_kind[kind] = out.by_kind.get(kind, 0) + 1
        c["signals"]["vision"] = score
        if label:
            c["vision_label"] = label
        key = "score_pre_events" if "score_pre_events" in c else "score"
        centre = out.centres[kind]
        base = c[key]
        c[key] = vision_blend(base, centred_rating(score, centre), style.vision_weight)
        # for sync_vision: the rating holds for this span only
        c["vision_rated"] = {
            "start": c["start"],
            "end": c["end"],
            "base": base,
            "blended": c[key],
            "kind": kind,
            "centre": round(centre, 2),
        }
        c.pop("vision_stale", None)
    return out


def _vision_stats(
    rater: VisionRater,
    *,
    rated: int,
    vision_s: float,
    pre_vision_s: float,
    rerank: bool = False,
) -> dict[str, Any]:
    """The vision step's cost for ``settings.vision`` (#909): the ratings made and taken from
    the cache, the step's seconds against the analysis before it, and the run's request
    timings and tokens summed (:attr:`VisionRater.stats`). Logs the one INFO line. ``rerank``:
    the run re-ranks an earlier analysis, so ``pre_vision_s`` is only the cached re-run and no
    share is logged."""
    st = rater.stats
    if rerank:
        logger.info("Vision: %d rated (%d cached) in %.0f s", rated, rater.hits, vision_s)
    else:
        share = 100 * vision_s / pre_vision_s if pre_vision_s > 0 else 0.0
        logger.info(
            "Vision: %d rated (%d cached) in %.0f s (%.0f%% of analyze before vision)",
            rated,
            rater.hits,
            vision_s,
            share,
        )
    return {
        **({"rerank": True} if rerank else {}),
        "rated": rated,
        "cached": rater.hits,
        "calls": int(st["calls"]),
        "vision_s": round(vision_s, 2),
        "pre_vision_s": round(pre_vision_s, 2),
        "grab_s": round(st["grab_s"], 2),
        "load_s": round(st["load_s"], 2),
        "prompt_s": round(st["prompt_s"], 2),
        "reply_s": round(st["reply_s"], 2),
        "prompt_tokens": int(st["prompt_tokens"]),
        "reply_tokens": int(st["reply_tokens"]),
        "model": rater.model,
        "provider": rater.provider_name,
    }


def _vision_batches(top: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """The candidates to rate in batches for ``--batch-size`` ``size``: those of ``size``
    sources each (sources in the order of their first candidate in ``top``, each one's
    candidates in band order, closest to the cut first). One batch, ``top`` in band order, for
    ``size`` 0 or ``size`` sources or fewer."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in top:
        groups.setdefault(c["source"], []).append(c)
    if size <= 0 or len(groups) <= size:
        return [top]
    parts = list(groups.values())
    return [[c for g in parts[i : i + size] for c in g] for i in range(0, len(parts), size)]


def analyze(
    target: Path | Sequence[Path],
    style: Style | AutoRequest,
    out_dir: Path,
    *,
    vision: bool | None = None,
    model: str | None = None,
    language: str | None = None,
    use_scenes: bool = True,
    max_candidates: int | None = None,
    vision_top: int | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    exclude: Sequence[str] = (),
    recursive: bool = False,
    recover: bool = False,
    layers: bool = False,
    skip: Collection[str] = (),
    earlier_total_s: float | None = None,
    detection: dict[str, Any] | None = None,
    batch_size: int = 0,
    visual_index: bool | None = None,
    ocr_det_skip: bool | None = None,
    crash_moments: bool = True,
    footage: category.FootageVision | None = None,
    vision_time_share: float | None = None,
    telemetry_fallback: bool = True,
) -> Path:
    """Run the full analysis and write ``out_dir/highlights.json``.

    The steps (:data:`cutpoint.progress.ANALYZE_STEPS`) run one after another, each
    model freed before the next one loads: transcription (audio, loudness, Whisper,
    scene cuts), with ``layers`` the event layers of ``cutpoint events`` (reactions,
    keyframes and crashes, see :func:`plan_layers`; events.json is written so a later
    ``cutpoint events`` reuses their caches), ranking (the events rescore the candidates)
    and the optional vision rating. highlights.json is written once, at the end, with
    transcript.txt and the other :data:`HELD_OUTPUTS` such as events.json and crashes.json
    (one commit): a failed or cancelled run leaves the previous ones untouched.

    A source whose read stops early (ffmpeg "Error during demuxing", an I/O
    error, or audio meaningfully shorter than the container) gets a loud
    warning and a ``read_errors`` entry in highlights.json. With ``recover``,
    analysis resumes after the bad region as an extra virtual source: the
    same file with an offset (no copy). Its candidates use absolute times in
    the original file.

    Args:
        target: Video file or folder, or a list of files and/or folders.
        style: Style preset, or an :class:`~cutpoint.styles.AutoRequest` (``--style auto``):
            the sources are probed before transcription and the style follows their
            footage category (:mod:`cutpoint.category`); highlights.json records the style
            used, ``settings.style_requested`` ``auto`` and the ``detection`` block.
        out_dir: Project directory.
        vision: True = rate the default edit and the moments below its cut (a warning if it is not
            available); False or None = no rating (opt-in, never automatic). For
            ``--style auto`` it also steers the footage question to the vision model about
            unsure sources and camera-tagged sources without telemetry: None asks quietly, True asks and warns of a missing consent,
            False does not ask.
        model: faster-whisper model name (default: medium on CUDA, small on CPU).
        language: Force transcription language (e.g. "en"); None = autodetect.
        use_scenes: Run scene detection.
        max_candidates: Max candidates per source before merging (None = auto).
        vision_top: Most moments sent to the vision model: the default edit and the moments
            below its cut (:func:`vision_band`) are cut to this many, closest to the cut first
            (None = the whole band).
        ollama_url: Ollama base URL.
        ollama_model: Ollama vision model name.
        exclude: Glob patterns (file name or full path) to skip.
        recursive: Search sub-folders of folder inputs.
        recover: Resume after read errors by seeking past the bad region.
        layers: Run the reactions, keyframes and crashes steps and report the steps
            (``CUTPOINT_PROGRESS``); False (``reanalyze``, ``360 analyze``): reuse
            events.json as it is.
        skip: Steps of :data:`cutpoint.progress.SKIPPABLE` left out.
        earlier_total_s: ``timings.total_s`` of the analysis this run re-ranks
            (:func:`reanalyze`): kept as ``total_s``, and this run's own time is recorded as
            ``timings.rerank_s``. None: this run's time is ``total_s``.
        detection: The ``detection`` block of the auto analysis this run re-ranks
            (:func:`reanalyze`, with the style it chose): recorded again as it is, and its
            category plans the layers. Nothing is detected again.
        batch_size: ``--batch-size`` (#109): Whisper is closed after every this many sources
            and loaded again for the next ones, and so are the event layers' models
            (:func:`~cutpoint.events.pipeline.run_events`); 0: one batch. The results are
            the same.
        visual_index: True = also write the frame embeddings (the keyframes step's CLIP visual
            layer) for gameplay and talk footage, for the assistant's visual search; they
            change no ranking. False or None = only the styles that always write them
            (action, sport, action-cam).
        ocr_det_skip: True = the HUD OCR skips the det gate of a region that has not
            changed since its last det call (``OcrParams.det_skip``), recorded as
            ``settings.ocr_det_skip`` for ``cutpoint events``; False or None = off.
        crash_moments: Make the crashes of crashes.json moments (:func:`add_crash_candidates`,
            every style but action-cam); False (``360 analyze``) leaves them out and records
            ``settings.crash_moments`` false, which :func:`reanalyze` keeps.
        footage: The caller's :class:`~cutpoint.category.FootageVision` for ``--style auto``
            (shared with its own detection; not closed here). None: one of this run.
        vision_time_share: The vision step's time cap, a share of the analysis before it
            (:func:`vision_budget`); None = ``vision_time_share`` in config.json.
        telemetry_fallback: An action-cam ``style`` (built-in or a style file over it) whose
            sources all lack camera telemetry (:func:`has_telemetry`) runs as ``action``
            instead, recorded as ``settings.style_requested`` and ``settings.style_fallback``
            with an ``action_cam_no_telemetry`` warning (#939). False (``360 analyze``): the
            style runs as given.

    Returns:
        Path of the written highlights.json.
    """
    with progress.run(ANALYZE_STEPS, use=layers) as steps:
        return _analyze(
            target,
            style,
            out_dir,
            steps,
            vision=vision,
            model=model,
            language=language,
            use_scenes=use_scenes,
            max_candidates=max_candidates,
            vision_top=vision_top,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            exclude=exclude,
            recursive=recursive,
            recover=recover,
            layers=layers,
            skip=skip,
            earlier_total_s=earlier_total_s,
            detection=detection,
            batch_size=batch_size,
            visual_index=bool(visual_index),
            ocr_det_skip=bool(ocr_det_skip),
            crash_moments=crash_moments,
            footage=footage,
            vision_time_share=vision_time_share,
            telemetry_fallback=telemetry_fallback,
        )


def _analyze(
    target: Path | Sequence[Path],
    style: Style | AutoRequest,
    out_dir: Path,
    steps: progress.Run,
    *,
    vision: bool | None,
    model: str | None,
    language: str | None,
    use_scenes: bool,
    max_candidates: int | None,
    vision_top: int | None,
    ollama_url: str | None,
    ollama_model: str | None,
    exclude: Sequence[str],
    recursive: bool,
    recover: bool,
    layers: bool,
    skip: Collection[str],
    earlier_total_s: float | None,
    detection: dict[str, Any] | None,
    batch_size: int,
    visual_index: bool,
    ocr_det_skip: bool,
    crash_moments: bool,
    footage: category.FootageVision | None,
    vision_time_share: float | None,
    telemetry_fallback: bool,
) -> Path:
    """:func:`analyze`, reporting to ``steps``."""
    t_start = time.perf_counter()
    steps.start("transcribe")
    sources = find_sources(target, exclude, recursive)
    # --style auto: probed before transcription, which needs the style's action-cam context, scene
    # setting and OCR prefetch; talk is told from general once the speech coverage is known.
    auto = style if isinstance(style, AutoRequest) else None
    probes: list[category.SourceProbe] = []
    cat: str | None = None  # the category whose layers run (None: the style's own)
    if auto is not None:
        probes, det = category.detect(sources, vision, ollama_url, ollama_model, shared=footage)
        name, fallback = category.choose_style(det, auto.emphasis)
        if det.pending:
            logger.info("Footage looks %s so far (%s)", category.label(det.category), det.reason)
        style = auto.style(name)
        cat = det.category if fallback is None else None
    elif detection is not None and not detection.get("fallback"):  # a re-run: as decided
        stored = detection.get("category")
        cat = stored if stored in category.CATEGORIES else None
    # action-cam without any telemetry: the action style, decided before the OCR prefetch, the
    # scene setting and the layer plan, which all follow the style
    style_fallback: dict[str, str] | None = None
    if (
        auto is None
        and detection is None
        and telemetry_fallback
        and style.kind == ACTION_CAM
        and not any(has_telemetry(s) for s in sources)
    ):
        style, style_fallback = _no_telemetry_fallback(style, sources)
    with (
        folders.output_dir(out_dir),  # a failed run leaves no empty project folder behind
        runstate.project_ready(out_dir),  # the run lock of a new project, now it exists
        cache.hold_outputs(out_dir, HELD_OUTPUTS) as held,
        # HUD OCR in the background while the GPU steps run; stopped before the folder goes
        _ocr_prefetch(
            style, sources, skip, out_dir, layers, cat, batch_size, ocr_det_skip
        ) as prefetch,
    ):
        # ---- 1. transcribe: audio, loudness, Whisper and scene cuts per source
        use_scenes = use_scenes and style.use_scenes
        cam_ctx = None
        if style.kind == ACTION_CAM:
            from cutpoint import actioncam  # noqa: PLC0415

            t0 = time.perf_counter()
            stale = out_dir / actioncam.FEATURES_DIR
            if stale.is_dir():
                for f in stale.glob("*.npz"):
                    cache.remove(f)  # held: removed only when the run completes
            durations = {str(s): probe(remap_source(s)).duration for s in sources}
            cam_ctx = actioncam.prepare(sources, durations, out_dir)
            telemetry_s = round(time.perf_counter() - t0, 2)
            _warn_missing_telemetry([s for s in sources if cam_ctx.telemetry[str(s)] is None])
        tr = Transcriber(model=model, language=language)

        def read_checked(src: Path, offset: float) -> Callable[..., Any] | None:
            """``analyze_source``'s hook: offer the source to the OCR prefetch, which reads
            only a whole, clean source while Whisper runs on the GPU (and, in a shared-decode
            mode, makes its scene proxy; the share is returned for the scene cuts)."""
            if prefetch is None:
                return None

            def offer(info: MediaInfo, rec: dict[str, Any] | None) -> Any:
                ok = not offset and rec is None and tr.device == "cuda"
                plan = None
                windows = edge_windows(style, info.duration)  # no full pass to share
                if ok and use_scenes and not windows and prefetch.mode is not None:
                    plan = _scene_plan(src, info.duration)
                return prefetch.offer(src, round(info.duration, 3), ok, plan)

            return offer

        def cpu_busy() -> bool:
            """The background HUD OCR (CPU-heavy) reads or is about to read a source: a source
            offered and not read yet (:attr:`~cutpoint.events.ocr_prefetch.OcrPrefetch.busy`)."""
            return prefetch is not None and prefetch.busy

        timings: dict[str, float] = {}
        if cam_ctx is not None:
            timings["telemetry_s"] = telemetry_s
        pending = auto is not None and det.pending  # the style's text signals wait for talk
        results: list[SourceResult] = []
        transcript_lines: list[str] = []
        progress.continuing("transcribe", _cached_transcripts(sources, tr), len(sources))
        for i, src in enumerate(sources):
            if batch_size and i and not i % batch_size:  # a batch boundary (--batch-size)
                tr.close()  # loaded again by the next transcript to make
                free_models()
                progress.continuing("transcribe", i, len(sources))
            steps.update(progress.fraction(i, len(sources)), src.name)

            shown = 0.0  # this source's share in the pct so far

            def report(done: float, i: int = i) -> None:
                """``done`` of source ``i`` in the step's pct, never lower than before (a
                recovered part's extract starts over at its own offset)."""
                nonlocal shown
                shown = max(shown, done)
                steps.update(progress.fraction(i + shown, len(sources)))

            offset, n_parts = 0.0, 0
            while True:
                logger.info(
                    "Analyzing %s%s",
                    src,
                    f" from {clock(offset)} (recovered part)" if offset else "",
                )
                # Action-cam candidates come from the feature grid, after the event layers.
                res = analyze_source(
                    src,
                    style,
                    tr,
                    use_scenes,
                    max_candidates,
                    timings,
                    offset=offset,
                    recover=recover,
                    pick=cam_ctx is None and not pending,
                    report=report if steps.active else None,
                    read_checked=read_checked(src, offset),
                    text=not pending,
                    # progress.json from the first transcript on, before the scene cuts
                    on_transcript=lambda: runstate.transcribed(out_dir, sources),
                    cpu_busy=cpu_busy,
                )
                results.append(res)
                name = f"{src.name}@{offset:.0f}" if offset else src.name
                transcript_lines.append(f"# {src}" + (f" (from {clock(offset)})" if offset else ""))
                transcript_lines.extend(transcript_text(name, res.segments) or ["(no speech)"])
                transcript_lines.append("")
                rec = res.read_error
                if not (recover and rec) or n_parts >= MAX_RECOVERED_PARTS:
                    break
                cdir = cache.source_dir(src, offset)
                resume = rec.get("resumed_at") or find_resume_point(
                    remap_source(src), rec["failure_time"], res.info.duration
                )
                if resume is None or resume <= offset:
                    logger.warning(
                        "%s: no readable point found after %s; rest of file skipped",
                        src.name,
                        clock(rec["failure_time"]),
                    )
                    break
                rec["resumed_at"] = resume
                rec["lost_s"] = round(resume - rec["failure_time"], 3)
                cache.save_json(cdir / READ_ERRORS_FILE, rec)
                logger.warning(
                    "%s: resuming at %s (skipping %.1f s of unreadable footage)",
                    src.name,
                    clock(resume),
                    rec["lost_s"],
                )
                offset, n_parts = resume, n_parts + 1
            runstate.transcribed(out_dir, sources)  # a source without audio: once analyzed
        # Written with highlights.json at the end; the event layers read it from memory.
        transcript = "\n".join(transcript_lines)
        tr.close()  # Whisper out of memory before the next step loads a model
        free_models()
        _warn_transcript_fallback(tr)
        detection_rec = detection  # highlights.json's record of an auto run (a re-run: as given)
        if auto is not None:  # the final decision, with the speech coverage
            det = category.decide(
                probes,
                category.source_speech(
                    sources, [(r.source, r.duration, r.segments) for r in results]
                ),
            )
            if pending:  # talk or general: this style's transcript signals and candidates
                # (otherwise the speech only adds the commentary hint: the style stays)
                name, fallback = category.choose_style(det, auto.emphasis)
                cat = det.category if fallback is None else None
                style = auto.style(name)
                for r in results:
                    if r.signals is not None:
                        text_signals(r.signals, style, r.source.name)
                        scene_signal(r.signals, style)  # a full pass if it weighs scenes
                        if r.signals.edge_cuts is None:  # cut by a full pass after all
                            r.stages.pop("scene_windows", None)
                        r.candidates = pick_candidates(r, style, max_candidates, timings)
            category.warn(det, style.name, fallback)
            detection_rec = det.record(auto.emphasis, style.name, fallback)
        steps.done()  # an error of the later steps never fails the transcription
        bad = [r for r in results if r.read_error]
        if bad:
            for r in bad:
                rec = r.read_error or {}
                logger.warning(
                    "READ ERROR SUMMARY: %s stopped at %s%s",
                    r.info.path.name,
                    clock(rec["failure_time"]),
                    f", resumed at {clock(rec['resumed_at'])}"
                    if rec.get("resumed_at")
                    else " (not recovered; use --recover)"
                    if not recover
                    else " (could not recover)",
                )
        hl: dict[str, Any] = {
            "version": 1,
            "style": style.name,
            "style_kind": style.kind,
            # The built-in style this one is based on (its own name for a built-in one), so
            # readers need no style library. Older files lack it.
            "style_base": style.base,
            **({"detection": detection_rec} if detection_rec is not None else {}),
            "created": None,
            "settings": {
                **({"mask_friends": style.mask_friends} if style.kind == ACTION_CAM else {}),
                "must_include_crash": style.must_include_crash,
                "whisper_model": tr.model_name,
                "whisper_device": tr.device,
                "scenes": use_scenes,
                "vision_requested": bool(vision),
                "vision_used": False,
                "vision": None,
                "vision_top": vision_top,
                "ollama_url": ollama_url,
                "ollama_model": ollama_model,
                "target_total": style.target_total,
                "weights": style.weights,
                # Which version of the style scored this (styles.style_hash): the file's
                # content hash for a user style. Recorded only; a re-rank does not compare it.
                "style_hash": style_hash(style),
                # The features this install can compute (install level; a source's own set,
                # which the scores divide by, is in ``features_by_source``).
                "features_available": sorted(features.available()),
                # Per source: the features its scores were divided by (#960). Differs from
                # features_available for an action-cam source without telemetry, or with
                # IMU-only telemetry (no GPS rows, no wind reading). Refilled after ranking.
                "features_by_source": _features_by_source(results),
                # Enough to re-run the same analysis (``reanalyze``, used by cutpoint events).
                # Absolute (not resolved: keeps drive letters, symlinks and the user's spelling)
                # so later commands find them from any working directory.
                "inputs": [
                    str(Path(t).absolute())
                    for t in ([target] if isinstance(target, Path) else target)
                ],
                "exclude": list(exclude),
                "recursive": recursive,
                "language": language,
                "max_candidates": max_candidates,
                "recover": recover,
                **({} if crash_moments else {"crash_moments": False}),  # 360 analyze
                **({"style_requested": AUTO} if detection_rec is not None else {}),
                # #939: an action-cam style without telemetry ran as `style` (action)
                **(
                    {"style_requested": style_fallback["from"], "style_fallback": style_fallback}
                    if style_fallback is not None
                    else {}
                ),
                # --ocr-det-skip: `cutpoint events` reads HUD text as this run did (only when on)
                **({"ocr_det_skip": True} if ocr_det_skip else {}),
            },
            "timings": timings,
            "suggested_top": 0,
            "sources": [
                {
                    "path": str(r.source),
                    "duration": round(r.info.duration, 3),
                    "width": r.info.width,
                    "height": r.info.height,
                    "fps": float(r.info.fps),
                    **(
                        {"offset": r.offset, "analyzed_s": round(r.duration, 3)}
                        if r.offset or r.read_error
                        else {}
                    ),
                    **({"read_errors": [r.read_error]} if r.read_error else {}),
                }
                for r in results
            ],
            "candidates": [],
        }

        # ---- 2-4. reactions, keyframes, crashes: the event layers (events.json)
        events_new = False
        if layers:
            plan = plan_layers(style, sources, skip, cat, visual_index)
            for step, reason in plan.skipped.items():
                steps.skip(step, reason)
            # ---- 5. dialogue: a text model reads the transcripts (open-list styles only).
            # Inside the event layers it runs before events.json is written; without them
            # (lite, every layer skipped) after, on the events.json there is.
            spoken: dict[str, list[dict[str, Any]]] = {}
            for r in results:
                spoken.setdefault(str(r.source), []).extend(r.segments)
            for segments in spoken.values():
                segments.sort(key=lambda s: s["start"])
            backend = _dialogue_backend(style, skip, steps, spoken, ollama_url, ollama_model)
            ran: list[bool] = []
            reader = (
                _dialogue_reader(backend, spoken, steps, timings, ran)
                if backend is not None
                else None
            )
            if plan.layers:
                # VLM windows cached by `events` follow them
                hl["candidates"] = _records(results, style)
                t0 = time.perf_counter()
                events_new = _run_layers(
                    out_dir,
                    plan,
                    hl,
                    steps,
                    llm="auto",  # profile discovery (activity.AUTO): any local Ollama, quietly
                    prefetch=prefetch,
                    transcript=transcript,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    batch_size=batch_size,
                    dialogue=(lambda layer1: reader(_audio_events(layer1))) if reader else None,
                    det_skip=ocr_det_skip,
                )
                timings["events_s"] = round(time.perf_counter() - t0, 2)
                free_models()
            if reader is not None and not ran:
                _dialogue_alone(
                    out_dir, hl, transcript, steps, reader, [str(r.source) for r in results]
                )

        # ---- 5. ranking: candidates rescored with the events
        steps.start("ranking")
        if cam_ctx is not None:
            from cutpoint import activity  # noqa: PLC0415

            if events_new:  # with this run's events.json and crashes.json
                cam_ctx = actioncam.prepare(sources, durations, out_dir)
            # the beauty / people features on the captions of the project's activity.json
            # (written by the crash step above; action-cam without one)
            with activity.use_prompts(activity.project_prompts(out_dir)):
                for r in results:
                    r.candidates = pick_candidates(r, style, max_candidates, timings, cam_ctx)
                    logger.info("%s: %d candidates", r.source.name, len(r.candidates))
            # a source's own set is known only once it is picked from
            hl["settings"]["features_by_source"] = _features_by_source(results)
        ev = rank_parts(
            hl,
            results,
            style,
            out_dir,
            steps,
            t_start=t_start,
            crash_moments=crash_moments,
            vision=vision,
            vision_top=vision_top,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            batch_size=batch_size,
            vision_time_share=vision_time_share,
            earlier_total_s=earlier_total_s,
        )
        hl["created"] = datetime.now().isoformat(timespec="seconds")
        hl["timings"]["sources"] = _source_timings(results, prefetch)
        took = round(time.perf_counter() - t_start, 2)
        if earlier_total_s is None:
            timings["total_s"] = took
        else:  # a re-rank: the analysis itself took what it took the first time
            timings["total_s"] = earlier_total_s
            timings["rerank_s"] = took
        # Text mode's line endings, as a plain write_text gave them; highlights.json last, so
        # the commit renames it last.
        cache.save_bytes(
            out_dir / "transcript.txt", transcript.replace("\n", os.linesep).encode("utf-8")
        )
        out = out_dir / "highlights.json"
        cache.save_json(out, hl)
        if ev:  # ranked with the events: a pre-events copy of an older analysis is stale
            cache.remove(out_dir / "highlights.pre-events.json")
        held.commit()  # every HELD_OUTPUTS file at once (a cancel in it waits for the last)
        logger.info("Wrote %d candidates to %s (%.1fs)", len(hl["candidates"]), out, took)
        report_highlights(out_dir, out)
        return out


def _records(parts: Sequence[CrashPart], style: Style) -> list[dict[str, Any]]:
    """The candidates of ``parts``, best first; ``source_index`` is the index into ``parts``."""
    found = [(c, i) for i, r in enumerate(parts) for c in r.candidates]
    found.sort(key=lambda ci: -ci[0].score)
    return [_candidate_record(c, i, style) for c, i in found]


def rank_parts(
    hl: dict[str, Any],
    parts: Sequence[CrashPart],
    style: Style,
    project: Path,
    steps: progress.Run,
    *,
    t_start: float,
    crash_moments: bool,
    vision: bool | None,
    vision_top: int | None,
    ollama_url: str | None,
    ollama_model: str | None,
    batch_size: int,
    vision_time_share: float | None,
    earlier_total_s: float | None,
) -> Any:
    """The ranking tail of an analysis: the candidates of ``parts`` (picked already) ranked
    into ``hl`` (a highlights.json object with its ``settings`` and ``timings``), in place.

    Runs inside the caller's ``ranking`` step (``steps``) and covers: the crash candidates of
    the project's crashes.json (``crash_moments``, any style but action-cam), the records, the
    bonus stage of the project's events.json (menu time, then :func:`rescore`: nominees,
    widening, bonus), the ``vision`` step (the rating of the default edit and the band
    below its cut, blended in and clamped; its time budget counts from ``t_start``, or
    ``earlier_total_s`` for a re-rank), the final order with the ids ``h001``... and
    ``settings.last_candidate_id``, and the visual pool. It writes no file.

    ``parts`` are one per entry of ``hl["sources"]``, in that order (a candidate's
    ``source_index`` is its part's position); a part without ``signals`` keeps its candidates
    but gives no grid. ``hl["timings"]`` gets ``vision_s``.

    Seams for a re-rank (not used yet): it numbers every candidate from ``h001`` (keeping the
    ids of unchanged spans would replace that), and the vision step asks the model (a cached-
    ratings-only rating would set :attr:`VisionRater.cache_only`).

    Returns:
        The events.json object the candidates were ranked with, or a falsy value without one.
    """
    timings = hl["timings"]

    if crash_moments and style.kind != ACTION_CAM:  # this run's crash step, `cutpoint crashes`
        from cutpoint.crash import load_crashes  # noqa: PLC0415

        add_crash_candidates(parts, style, load_crashes(project))
    hl["candidates"] = _records(parts, style)
    ev = None
    if style.kind != ACTION_CAM:  # action-cam: the events are features of its scorer
        ev = cache.load_json(project / "events.json")
    grids = [
        GridPart(r.source, i, r.offset, r.duration, r.signals)
        for i, r in enumerate(parts)
        if r.signals is not None
    ]
    if ev:
        drop_menu_time(hl, ev)
        changed = rescore(hl, ev, project, grids)
        logger.info("Rescored %d of %d candidates from events.json", changed, len(hl["candidates"]))

    # ---- 6. vision: the rating of the default edit and the moments below its cut, blended into
    # their scores
    cands = hl["candidates"]
    rater: VisionRater | None = None
    bad_url: ValueError | None = None
    if vision and cands:
        try:
            rater = VisionRater(
                **{k: v for k, v in {"url": ollama_url, "model": ollama_model}.items() if v}
            )
        except ValueError as err:  # a bad port in OLLAMA_HOST / ollama_url
            bad_url = err
    if not vision:
        steps.skip("vision", "vision is off (Settings, config.json or --vision)")
    elif not cands:
        steps.skip("vision", "no candidates to rate")
    elif rater is None:
        message = f"vision model not usable ({bad_url}); the vision step is skipped"
        logger.warning("%s", message)
        jsonout.warn("vision_unavailable", message)
        steps.skip("vision", message)
    elif rater.consent_needed:
        # a provider the user did not agree to (a hosted one, or an Ollama on another
        # machine): no request, no cached ratings either
        steps.skip("vision", rater.warn_consent())
    else:
        from cutpoint.visual_pool import is_pool  # noqa: PLC0415

        # the default edit and the band below its cut, from the ranking before vision
        ranked = sorted(
            (c for c in cands if not c.get("must_include") and not is_pool(c)),
            key=scorer.edit_rank_key,
        )
        picks = default_fit.fit_target(
            sorted((c for c in cands if not is_pool(c)), key=scorer.edit_rank_key),
            style.target_total,
        )
        order, cut, k = vision_band(ranked, picks, vision_top)
        in_edit = {id(c) for c in picks}
        # the moments above the cut the edit skipped (too long or overlapping a pick): no
        # one's neighbours in the clamp, as long as they keep this span
        for c in ranked[:cut]:
            if id(c) not in in_edit:
                c["vision_skipped"] = [c["start"], c["end"]]
        # action-cam: worded for the project's activity.json (the default without one)
        wording = None
        if style.base == ACTION_CAM:
            from cutpoint import activity  # noqa: PLC0415

            wording = activity.project_prompts(project)
        # available() first: it resolves a hosted provider's unset model, the cache key.
        # Offline, that model stays the placeholder and its cached ratings are not found.
        online, n_cached = rater.available(), 0
        if not online:  # Ollama down: the ratings already cached, no requests (#413)
            rater.cache_only = True
            n_cached = sum(
                rater.cached(
                    remap_source(Path(c["source"])),
                    c["start"],
                    c["end"],
                    style.base,
                    activity=wording,
                    transcript=c.get("transcript"),
                    laughter=heard_laughter(c),
                )
                is not None
                for c in order
            )
        if online or n_cached:
            if not online:
                n_none = len(order) - n_cached
                message = (
                    "vision rating requested but Ollama/model unavailable; using "
                    f"{n_cached} cached rating{'s' if n_cached != 1 else ''}, "
                    f"{n_none} candidate{'s' if n_none != 1 else ''} had no cached rating"
                )
                logger.warning("%s", message)
                jsonout.warn("vision_unavailable", message)
            with steps.step("vision"):
                t0 = time.perf_counter()
                pre_vision_s = t0 - t_start
                share = (
                    config.vision_time_share() if vision_time_share is None else vision_time_share
                )
                budget_s, budget_base = vision_budget(share, pre_vision_s, earlier_total_s)
                n_edit = sum(id(c) in in_edit for c in order)
                logger.info(
                    "Vision: rating %d moments (%d of the default edit, %d below the cut at "
                    "%d) of %d candidates",
                    len(order),
                    n_edit,
                    len(order) - n_edit,
                    cut,
                    len(cands),
                )
                outcome = _rate_vision(
                    order, rater, style, wording, batch_size, deadline=t0 + budget_s
                )
                if ev:  # the blended analysis scores plus the event boosts again
                    hl["settings"]["vision"] = {}  # rescore_events records the clamp here
                    rescore(hl, ev, project, grids)
                    n_clamped = hl["settings"]["vision"].get("clamped", 0)
                else:  # a rated moment stays between its unrated neighbours
                    n_clamped = scorer.apply_vision_clamp(
                        cands,
                        [c["vision_rated"]["base"] if "vision_rated" in c else None for c in cands],
                    )
                timings["vision_s"] = round(time.perf_counter() - t0, 2)
                if outcome.capped:
                    logger.info(
                        "Vision: time cap reached after %d calls (%.0f s, budget %.0f s = "
                        "%d%% of %.0f s); %d moments near the cut not rated",
                        int(rater.stats["calls"]),
                        timings["vision_s"],
                        budget_s,
                        round(100 * share),
                        pre_vision_s if earlier_total_s is None else earlier_total_s,
                        outcome.skipped_by_cap,
                    )
                hl["settings"]["vision_used"] = outcome.used
                hl["settings"]["vision"] = {
                    "requested": len(order),
                    "failed": outcome.failed - rater.missed,
                    **({} if online else {"uncached": rater.missed}),
                    **_vision_stats(
                        rater,
                        rated=len(order) - outcome.failed - outcome.skipped_by_cap,
                        vision_s=timings["vision_s"],
                        pre_vision_s=pre_vision_s,
                        rerank=earlier_total_s is not None,
                    ),
                    "cut": cut,
                    "band_k": k,
                    "rated_by_kind": outcome.by_kind,
                    "centre": {
                        kind: (round(outcome.centres[kind], 2) if kind in outcome.centres else None)
                        for kind in ("speech", "frames")
                    },
                    "share": share,
                    "budget_s": round(budget_s, 2),
                    "budget_base": budget_base,
                    "capped": outcome.capped,
                    "skipped_by_cap": outcome.skipped_by_cap,
                    "clamped": n_clamped,
                }
        else:
            message = "vision rating requested but Ollama/model unavailable; continuing without it"
            logger.warning("%s", message)
            jsonout.warn("vision_unavailable", message)
            steps.skip("vision", "Ollama or its vision model is not reachable")

    cands = hl["candidates"]
    cands.sort(
        key=lambda c: (not c.get("must_include"), *scorer.rank_key(c), *scorer.tie_key(c))
    )  # must-include first
    suggested = len(default_fit.fit_target(cands, style.target_total))
    hl["candidates"] = [{"id": f"h{rank:03d}", **c} for rank, c in enumerate(cands, 1)]
    hl["suggested_top"] = suggested
    hl["settings"]["last_candidate_id"] = len(cands)
    from cutpoint import visual_pool  # noqa: PLC0415

    if visual_pool.is_pool_style(style):
        # visual pool moments: after the scored ones, outside suggested_top
        hl["candidates"] += _visual_pool(project, parts, style, hl)
    return ev


def report_highlights(out_dir: Path, highlights: Path) -> None:
    """``--json`` outputs of an analysis: the files written and what highlights.json holds
    (``edit`` calls it again after ``events`` re-ranked it; the later values replace these)."""
    if not jsonout.active():
        return
    saved = cache.load_json(highlights)
    saved = saved if isinstance(saved, dict) else {}
    jsonout.output("highlights", highlights)
    jsonout.output("transcript", out_dir / "transcript.txt")
    jsonout.output("candidates", len(saved.get("candidates") or []))
    from cutpoint import visual_pool  # noqa: PLC0415

    if visual_pool.project_has_pool(saved):  # absent for the other styles
        cands = saved.get("candidates") or []
        jsonout.output("visual_candidates", sum(visual_pool.is_pool(c) for c in cands))
    if "suggested_top" in saved:
        jsonout.output("suggested_top", saved["suggested_top"])
    jsonout.output("style", saved.get("style"))
    if isinstance(saved.get("detection"), dict):  # --style auto
        jsonout.output("detection", saved["detection"])


def _recorded_style(hl: dict[str, Any]) -> Style:
    """The style of the analysis ``hl``, with the ``--must-include-crash`` it recorded.

    A style no longer in the library (a deleted user style) degrades to its recorded built-in
    base (:func:`cutpoint.events.scoring.recorded_style`).

    Raises:
        ValueError: If the style is not in the library and ``hl`` records no built-in base.
    """
    style = recorded_style(hl)
    at = (hl.get("settings") or {}).get("must_include_crash")
    if at is not None:  # an `analyze --must-include-crash`
        style = replace(style, must_include_crash=float(at))
    return style


def reanalyze(project: Path) -> Path:
    """Re-run ``analyze`` with the inputs and style recorded in highlights.json.

    Everything expensive is cached (audio, transcript, telemetry), so this is
    quick. ``cutpoint events`` calls it for action-cam projects, whose events feed the
    feature grid itself rather than a score boost. ``timings.total_s`` stays the earlier
    analysis's; the re-rank's own time is ``timings.rerank_s``.

    Raises:
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If highlights.json is unreadable, not valid JSON or records
            no analysis, or its style is not in the library and it records no built-in
            base (:func:`_recorded_style`).
    """
    from cutpoint.pano.project import PANO_FILE  # noqa: PLC0415

    try:
        hl = cache.read_json(
            project / "highlights.json", hint="; re-run `cutpoint analyze` to rewrite it"
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"{project / 'highlights.json'} not found") from None
    # e.g. an empty {} or a list: a clear error, not a KeyError or AttributeError traceback
    if not isinstance(hl, dict) or "style" not in hl or "sources" not in hl:
        raise ValueError(
            f"{project / 'highlights.json'} records no analysis; run 'cutpoint analyze' again"
        )
    st = hl.get("settings", {})
    inputs = st.get("inputs") or [s["path"] for s in hl["sources"]]
    pano = (project / PANO_FILE).is_file()
    style = _recorded_style(hl)
    requested = st.get("style_requested")
    if isinstance(st.get("style_fallback"), dict) and isinstance(requested, str):
        # an action-cam style that fell back to action (#939): asked for again, so the
        # telemetry probe decides as it did (or picks it up once the sources have it)
        from cutpoint.styles import get_style, learned_styles  # noqa: PLC0415

        try:
            style = replace(get_style(requested), must_include_crash=style.must_include_crash)
        except ValueError:  # a deleted user style: the recorded one, without the probe
            if requested in learned_styles():  # there but unusable (broken file): its error
                raise
            logger.info(
                "Style %r is not in the style library (deleted?); using %r", requested, style.name
            )
    total = (hl.get("timings") or {}).get("total_s")
    # --style auto: the style it chose, and its decision kept (never detected again)
    detection = hl.get("detection") if st.get("style_requested") == AUTO else None
    return analyze(
        [Path(p) for p in inputs],
        style,
        project,
        # opt-in: only if the earlier run asked (older projects record no request: off)
        vision=st.get("vision_requested") is True,
        vision_top=st.get("vision_top"),
        ollama_url=st.get("ollama_url"),
        ollama_model=st.get("ollama_model"),
        model=st.get("whisper_model"),
        language=st.get("language"),
        use_scenes=bool(st.get("scenes", True)),
        max_candidates=st.get("max_candidates"),
        exclude=st.get("exclude") or [],
        recursive=bool(st.get("recursive")),
        recover=bool(st.get("recover")),
        # the analysis time stays the first run's; this re-rank's is timings.rerank_s
        earlier_total_s=float(total) if isinstance(total, int | float) else None,
        detection=detection if isinstance(detection, dict) else None,
        ocr_det_skip=st.get("ocr_det_skip") is True,
        # off for 360 projects (pano.json: also one analysed before crash_moments was recorded)
        crash_moments=st.get("crash_moments") is not False and not pano,
        telemetry_fallback=not pano,  # 360 projects stay action-cam
    )


def reanalyze_if_reachable(project: Path) -> Path | None:
    """:func:`reanalyze`, unless a source of the project is offline.

    ``analyze`` refuses a missing input, and without it the source would drop out
    of highlights.json; ``crashes`` and ``events`` already skipped it, so they
    leave highlights.json unchanged instead of failing. Likewise when analyze would
    not find an input, or a recorded source under its recorded path, even through
    ``source_remap``: re-analysing would fail, or drop or re-key the source.

    Returns:
        Path of highlights.json, or None if skipped.
    """
    hl = cache.load_json(project / "highlights.json")
    if not isinstance(hl, dict):  # missing or not an object: reanalyze below reports it
        hl = {}
    offline = 0
    for path in dict.fromkeys(s["path"] for s in hl.get("sources", [])):
        try:
            cache.source_dir(Path(path))
        except OSError:
            offline += 1
    if offline:
        logger.warning(
            "%d source(s) not reachable; highlights.json left unchanged, "
            "rerun once the footage is back",
            offline,
        )
        return None
    # Run analyze's own input search (through ``source_remap`` for a moved input): an
    # input it cannot find would fail it, and a recorded source it does not find under
    # the same spelling would drop out (e.g. moved out of a folder input that still
    # exists) or be re-keyed (an older project that recorded the remapped copy).
    st = hl.get("settings", {})
    srcs = [s["path"] for s in hl.get("sources", [])]
    found: set[str] = set()
    missing = 0
    for p in dict.fromkeys(st.get("inputs") or srcs):
        try:
            batch = _find_input(Path(p), st.get("exclude") or [], bool(st.get("recursive")))
        except FileNotFoundError:
            missing += 1
            continue
        found.update(os.path.normcase(str(f)) for f in batch)
    missing += sum(os.path.normcase(str(Path(p))) not in found for p in dict.fromkeys(srcs))
    if missing:
        logger.warning(
            "%d path(s) not found where analyze recorded them, even through source_remap; "
            "highlights.json left unchanged, rerun once the footage is back",
            missing,
        )
        return None
    return reanalyze(project)
