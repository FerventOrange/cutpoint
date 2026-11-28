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
import gc
import importlib.util
import logging
import os
import sys
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from cutpoint import (
    backends,
    cache,
    category,
    default_fit,
    features,
    folders,
    intervals,
    jsonout,
    memguard,
    progress,
    reanalyze,
    repeats,
    runstate,
    scorer,
    selfcheck,
    tsrepair,
    vision_step,
)
from cutpoint import scenes as scene_mod
from cutpoint.config import remap_source, unc_path
from cutpoint.crash_moments import CrashPart, add_crash_candidates
from cutpoint.dsp import conv_same, gauss_kernel, gauss_smooth
from cutpoint.errors import LowMemoryError
from cutpoint.events import paths as event_paths
from cutpoint.events.layers import overlay_kinds
from cutpoint.events.menu_spans import drop_menu_time
from cutpoint.events.scoring import (
    Seed,
    number_candidates,
    rescore,
)
from cutpoint.events.setup import COMPONENTS, EVENTS_MODULES
from cutpoint.ffmpeg_utils import (
    READ_ERROR_RE,
    READ_PROBE_S,
    MediaInfo,
    ProgressResult,
    extract_audio,
    momentary_loudness,
    probe,
    read_window_ok,
    source_tracks,
    wav_duration,
)
from cutpoint.progress import ANALYZE_STEPS, LABELS, SKIPPABLE
from cutpoint.projectfiles import HIGHLIGHTS, PRE_EVENTS
from cutpoint.sources import READ_ERRORS_FILE, load_read_errors, source_segments
from cutpoint.styles import (
    ACTION_CAM,
    ACTION_STYLES,
    AUTO,
    PROFANITY,
    AutoRequest,
    Style,
    style_hash,
)
from cutpoint.telemetry import has_telemetry
from cutpoint.transcribe import (
    Segment,
    Transcriber,
    cached_transcript,
    default_model,
    filter_hallucinations,
    flat_markers,
    flat_words,
    is_fallback,
    keyword_hits,
    laughter_hits,
    record_transcript,
    shift_segments,
    split_markers,
    transcript_name,
)
from cutpoint.utils import (
    clock,
    find_videos,
    footage_context,
    matches_glob,
    slugify,
    title_slug,
)

if TYPE_CHECKING:
    from cutpoint.events.ocr_prefetch import OcrPrefetch

logger = logging.getLogger(__name__)

DT = 0.1  # analysis grid step (s), matches ebur128's 10 Hz momentary output
PROFANE = {p.rstrip("*") for p in PROFANITY}  # names as keyword_hits reports them
PROFANITY_CUE_S = 3.0  # a profane word counts only if another cue is this close
PROFANITY_CUE_MIN = 0.8  # minimum weight of a keyword that counts as a cue
# Delivery cue of a hype word (#1071, styles with ``delivery_cue``): within DELIVERY_CUE_S of it,
# laughter, a non-profane keyword other than itself of at least PROFANITY_CUE_MIN, or a loudness
# jump: the loudness grid's highest from DELIVERY_PRE_S before the word to DELIVERY_POST_S after
# it at least DELIVERY_JUMP above its median over the DELIVERY_BASE_S before that (the grid is
# already the source's own robust z-score, so the jump is relative to the source), or at least
# DELIVERY_LOUD (the loudness that lets a profane word count alone). A hype word with none keeps
# DELIVERY_FLAT of its weight: said flat it is still talk about the moment, so it can break a tie
# and name a clip, but no longer outranks a loud one (0.3: what a secondary hit adds under the
# keyword cap, :func:`keyword_signal`). On the 3 h stream of #1071, 89% of the "oh my god" hits
# had no such cue.
DELIVERY_CUE_S = PROFANITY_CUE_S
DELIVERY_PRE_S, DELIVERY_POST_S, DELIVERY_BASE_S = 1.0, 2.0, 9.0
DELIVERY_JUMP = 0.2
DELIVERY_LOUD = 0.5
DELIVERY_FLAT = 0.3
PICK_FLOOR = 0.25  # a peak below this share of the source's reference is not a candidate ...
LONG_LIST_FLOOR = 0.5  # ... and, past the usual count (:func:`usual_count`), not below this share
# The reference of those shares (:func:`pick_reference`, #1085 A104): the strongest window peak
# of the whole source, over all its recovered parts, but at most the second strongest over
# LONG_LIST_FLOOR, so the source's second moment always clears the long-list floor. One spike
# no longer hides the source's other moments, and where the strongest is not a spike (most
# footage) the reference is the strongest peak as before, so nothing changes there.
# Nor is a peak at or below this share of the style's weight sum (a 5-point score, :func:`
# cutpoint.features.to_score`), whatever the reference: what keeps a dull source quiet. Was a
# fixed 0.05 of the raw sum: the same for styles whose weights sum to 1, slightly different
# for `sport` (1.1) and custom styles whose weights do not (#1085 A104)
PICK_MIN = 0.05
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
# Written next to the analysis WAV once it is read on the container's clock (a gap in the
# timestamps is silence, ffmpeg_utils.AUDIO_CLOCK, #1056); a WAV without it is checked once
CLOCK_FILE = "audio16k-clock.json"
# The caches made from the analysis WAV and, last, the WAV, removed when it mixed other tracks
# or left out a gap
AUDIO_CACHES = ("loudness.json", "transcript-*.json", "fallback-transcript-*.json",
                "events-audio-*.json", "voice-*.npz", CLOCK_FILE, "audio16k.wav")  # fmt: skip
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
    HIGHLIGHTS,
    PRE_EVENTS,  # deleted once ranked with the events
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
def _impulse_signal(n: int, events: list[tuple[float, float]], sigma_s: float = 1.0) -> np.ndarray:
    """Place weighted impulses, blur them (peak height == weight), then saturate to 0-1."""
    x = np.zeros(n)
    for t, w in events:
        i = int(t / DT)
        if 0 <= i < n:
            x[i] += w
    _, k = gauss_kernel(sigma_s, DT, normalise=False)
    return 1.0 - np.exp(-conv_same(x, k))


def loudness_signal(lufs: np.ndarray) -> np.ndarray:
    """Positive robust z-score of loudness, mapped to 0-1 (z=4 -> ~1)."""
    x = np.maximum(lufs, -70.0)  # silence floor
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    z = np.clip((x - med) / max(mad, 1.0), 0, None)
    return np.clip(gauss_smooth(z, 0.4, dt=DT, pad="zero") / 4.0, 0, 1)


def continuity_signal(n: int, words: list[dict[str, Any]], window_s: float = 10.0) -> np.ndarray:
    """Fraction of a sliding window covered by spoken words (0-1)."""
    talk = np.zeros(n)
    for w in words:
        talk[int(w["start"] / DT) : int(w["end"] / DT) + 1] = 1.0
    k = int(window_s / DT)
    return np.clip(conv_same(talk, np.ones(k) / k) * 1.25, 0, 1)


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
    radius, k = gauss_kernel(sigma_s, DT, normalise=False)
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


def damp_keywords(
    kw_hits: list[tuple[float, float, str]], span_s: float
) -> list[tuple[float, float, str]]:
    """``kw_hits`` of one source (or part) ``span_s`` seconds long with each weight times the
    repeat factor of its phrase there (:func:`cutpoint.repeats.repeat_factor`): a phrase said
    often counts less each time, a rare one keeps its weight (#1071)."""
    factors = repeats.repeat_factors((p for _, _, p in kw_hits), span_s)
    return [(t, w * factors[p], p) for t, w, p in kw_hits]


def loudness_jump(loudness: np.ndarray, t: float) -> bool:
    """Whether the loudness grid jumps at ``t`` (seconds): its highest over
    ``[t - DELIVERY_PRE_S, t + DELIVERY_POST_S]`` is at least :data:`DELIVERY_LOUD`, or at least
    :data:`DELIVERY_JUMP` above its median over the :data:`DELIVERY_BASE_S` before that."""
    n = len(loudness)
    if not n:
        return False
    i = min(n - 1, max(0, int(t / DT)))
    lo = max(0, i - int(DELIVERY_PRE_S / DT))
    peak = float(loudness[lo : i + int(DELIVERY_POST_S / DT) + 1].max())
    base = loudness[max(0, lo - int(DELIVERY_BASE_S / DT)) : lo]
    return peak >= DELIVERY_LOUD or peak - (float(np.median(base)) if base.size else 0.0) >= (
        DELIVERY_JUMP
    )


def gate_delivery(
    kw_hits: list[tuple[float, float, str]],
    laugh_hits: list[tuple[float, float, str]],
    loudness: np.ndarray,
) -> list[tuple[float, float, str]]:
    """``kw_hits`` with each non-profane hit that has no delivery cue near it (laughter,
    another strong non-profane keyword, a loudness jump: :data:`DELIVERY_CUE_S`,
    :func:`loudness_jump`) at :data:`DELIVERY_FLAT` of its weight; profane hits are left to
    :func:`gate_profanity`. A cue keyword is a different phrase: a filler said twice in a row is
    no cue for itself. The cues are judged on the weights given, so the result does not depend
    on the order of the hits."""
    strong = [(t, p) for t, w, p in kw_hits if p not in PROFANE and w >= PROFANITY_CUE_MIN]
    laughs = np.asarray([t for t, _, _ in laugh_hits], dtype=float)
    out = []
    for t, w, p in kw_hits:
        if p not in PROFANE and not (
            (laughs.size and float(np.min(np.abs(laughs - t))) <= DELIVERY_CUE_S)
            or any(q != p and abs(u - t) <= DELIVERY_CUE_S for u, q in strong)
            or loudness_jump(loudness, t)
        ):
            w *= DELIVERY_FLAT
        out.append((t, w, p))
    return out


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
    if not _purge_audio(src, cdir, "mixed other tracks"):
        return
    if want is None:
        (cdir / MIX_FILE).unlink(missing_ok=True)
    else:
        cache.save_json(cdir / MIX_FILE, {"tracks": want})


def _purge_audio(src: Path, cdir: Path, why: str, extra: tuple[str, ...] = ()) -> bool:
    """Remove the analysis WAV of ``src`` from ``cdir`` with every cache made from it
    (:data:`AUDIO_CACHES`) and the ``extra`` files, the WAV last; ``why`` is what the INFO line
    says was wrong with the old audio. A file that cannot be removed (open in another program)
    is named in a WARNING; the WAV is then kept too, so the old WAV and the caches left are
    used together meanwhile. Returns whether every file is gone."""
    patterns = (*AUDIO_CACHES[:-1], *extra, AUDIO_CACHES[-1])
    stale = [f for pattern in patterns for f in cdir.glob(pattern)]
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
        return False
    if stale:  # the WAV (removed last) is gone: extracted again
        logger.info("%s: the cached analysis audio %s; extracting it again", src.name, why)
    return True


def _audio_clock(src: Path, info: MediaInfo, offset: float, cdir: Path) -> None:
    """Make the cached analysis WAV in ``cdir`` one read on the container's clock (#1056).

    A WAV made before :data:`CLOCK_FILE` existed left out the stretch where a forward jump in
    the timestamps (under tsrepair's limit) was: it is shorter than the source, its read-error
    record (if any) has no ffmpeg message, and the transcript after the jump is early. Such a
    WAV is removed with every cache made from it and the length-only record, so it is
    extracted again once. A WAV of full length, or one cut short by a read error ffmpeg
    reported, is kept; either way :data:`CLOCK_FILE` is written, so the check is made once.
    ``info`` is the probe of the file the WAV is read from (the copy of a repaired source).
    """
    wav = cdir / "audio16k.wav"
    if not wav.exists() or (cdir / CLOCK_FILE).exists():
        return
    short = (info.duration - offset) - wav_duration(wav)
    saved = load_read_errors(cdir)
    if short > SHORTFALL_FLOOR_S and not (saved and saved.get("messages")):
        why = f"left out {short:.1f} s where the timestamps jump"
        _purge_audio(src, cdir, why, extra=(READ_ERRORS_FILE,))
        return  # gone: its extract writes the record; kept: the next run checks again
    _write_clock(cdir)


def _write_clock(cdir: Path) -> None:
    """Record that the WAV in ``cdir`` was read on the container's clock."""
    cache.save_json(cdir / CLOCK_FILE, {"version": 1})


SOURCE_STAGES = ("probe_s", "audio_extract_s", "loudness_s", "transcribe_s", "scenes_s")


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
    from cutpoint.ffmpeg_utils import tmp_files  # noqa: PLC0415 - one function; keeps the import block

    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(EXTRACT_POLL_S):
            try:  # extract_audio's unique temp file, renamed when complete
                size = max((p.stat().st_size for p in tmp_files(wav, own=True)), default=0)
            except OSError:  # renamed while listed
                continue
            if size == 0:  # not created yet
                continue
            report(min(1.0, max(0, size - WAV_HEADER) / (WAV_BYTES_PER_S * seconds)))

    thread = threading.Thread(target=poll, name="extract-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


# --------------------------------------------------------------------------- windowing
def _pick_windows(
    src: Path,
    combined: np.ndarray,
    sig: dict[str, np.ndarray],
    style: Style,
    duration: float,
    max_candidates: int,
    usual: int | None = None,
    ref: float | None = None,
) -> list[Candidate]:
    """Windows around the highest peaks of ``combined``, strongest first, at most
    ``max_candidates``. A peak must be above :data:`PICK_MIN` of the style's weight sum and
    reach :data:`PICK_FLOOR` of ``ref``, the source's reference (:func:`pick_reference`;
    None: that of ``combined`` alone), and once ``usual`` windows are picked (None: never)
    :data:`LONG_LIST_FLOOR` of it."""
    if ref is None:
        ref = _reference(_top_peaks(combined, style, duration))
    least = PICK_MIN * features.divisor(style.weights)
    score = combined.copy()
    out: list[Candidate] = []
    while len(out) < max_candidates:
        i = int(score.argmax())
        peak = float(score[i])
        floor = LONG_LIST_FLOOR if usual is not None and len(out) >= usual else PICK_FLOOR
        if peak <= least or peak < floor * ref:
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


def _top_peaks(combined: np.ndarray, style: Style, duration: float) -> list[float]:
    """The two strongest window peaks of one part's ``combined`` (:func:`_pick_windows`
    without the relative floors)."""
    return [c.peak for c in _pick_windows(Path(), combined, {}, style, duration, 2, ref=0.0)]


def pick_reference(parts: Sequence[PickPart], style: Style) -> float:
    """The reference of :data:`PICK_FLOOR` and :data:`LONG_LIST_FLOOR` for the parts of one
    source (a source read as recovered parts: all of them, so they share one floor): its
    strongest window peak, at most its second strongest over :data:`LONG_LIST_FLOOR`; 0
    without any above :data:`PICK_MIN`."""
    return _reference(
        [
            p
            for part in parts
            if part.signals is not None
            for p in _top_peaks(scorer.combined(part.signals.sig, style), style, part.duration)
        ]
    )


def _reference(peaks: list[float]) -> float:
    """:func:`pick_reference` of the window ``peaks`` (0: none)."""
    top = sorted(peaks, reverse=True)
    if len(top) < 2:
        return top[0] if top else 0.0
    return min(top[0], top[1] / LONG_LIST_FLOOR)


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


class _SnapIndex:
    """The words, sentence edges and speech stretches of one part's ``segments``, built once
    for every candidate :func:`_snap` snaps there (A685). Every lookup returns what the scan
    over the segments gave, to the float."""

    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments
        self.words = flat_words(segments)
        raw_starts = [w["start"] for w in self.words]
        raw_ends = [w["end"] for w in self.words]
        self.starts = sorted(raw_starts)
        self.ends = sorted(raw_ends)
        self.starts_sorted = raw_starts == self.starts
        self.ends_sorted = raw_ends == self.ends

    @cached_property
    def sentences(self) -> tuple[list[float], list[float]]:
        starts, ends = sentence_bounds(self.segments)
        return sorted(starts), sorted(ends)

    @cached_property
    def _speech(self) -> tuple[list[float], list[float]]:
        spans = sorted(
            (s["start"] - 0.3, s["end"] + 0.3)
            for s in self.segments
            if s.get("text") and s["start"] - 0.3 <= s["end"] + 0.3
        )
        los: list[float] = []
        his: list[float] = []
        for lo, hi in spans:
            if his and lo <= his[-1]:
                his[-1] = max(his[-1], hi)
            else:
                los.append(lo)
                his.append(hi)
        return los, his

    def in_speech(self, t: float) -> bool:
        los, his = self._speech
        i = bisect_right(los, t) - 1
        return i >= 0 and t <= his[i]

    def gap_before(self, t: float) -> float:
        i = bisect_right(self.ends, t)
        return t - self.ends[i - 1] if i else 1.0

    def gap_after(self, t: float) -> float:
        i = bisect_left(self.starts, t)
        return self.starts[i] - t if i < len(self.starts) else 1.0

    def longest_pause(self, lo: float, hi: float) -> tuple[float, float] | None:
        if not self.ends_sorted:
            return _longest_pause(self.words, lo, hi)
        first = bisect_left(self.ends, lo)
        last = min(bisect_right(self.ends, hi), len(self.words) - 1)
        return _longest_pause(self.words[first : last + 1], lo, hi)


def _snap_sentences(
    c: Candidate, segments: list[Segment], duration: float, index: _SnapIndex | None = None
) -> None:
    """Move boundaries that fall inside speech to sentence ends, else the longest pause.

    Padding never exceeds half the silence next to the boundary, so it cannot
    reach into a neighbouring word. ``index`` is the :class:`_SnapIndex` of ``segments``
    when the caller snaps many candidates.
    """
    ix = index or _SnapIndex(segments)
    starts, ends = ix.sentences
    if ix.in_speech(c.start):
        i = bisect_right(starts, c.start)  # starts[:i] are <= c.start, starts[i:] are after
        t = None
        if i and starts[i - 1] >= c.start - 12:
            t = starts[i - 1]
        elif i < len(starts) and starts[i] <= c.start + 6:
            t = starts[i]
        if t is None:
            pause = ix.longest_pause(c.start - 12, c.start + 6)
            t = pause[1] if pause else c.start
        c.start = max(0.0, t - min(0.2, ix.gap_before(t) / 2))
    if ix.in_speech(c.end):
        i = bisect_left(ends, c.end)  # ends[:i] are before c.end, ends[i:] are >= c.end
        t = None
        if i < len(ends) and ends[i] <= c.end + 15:
            t = ends[i]
        elif i and ends[i - 1] >= c.end - 6 and ends[i - 1] > c.start:
            t = ends[i - 1]
        if t is None:
            pause = ix.longest_pause(c.end - 6, c.end + 15)
            t = pause[0] if pause else c.end
        c.end = min(duration, t + min(0.4, ix.gap_after(t) / 2))


def _snap(
    c: Candidate,
    cuts: list[float],
    segments: list[Segment],
    duration: float,
    style: Style,
    index: _SnapIndex | None = None,
) -> None:
    """Snap to nearby scene cuts, then sentence edges (long-form), then word edges.

    ``index`` is the :class:`_SnapIndex` of ``segments``: pass one when snapping many
    candidates of the same part."""
    ix = index or _SnapIndex(segments)
    for attr in ("start", "end"):
        t = getattr(c, attr)
        near = [x for x in cuts if abs(x - t) <= scene_mod.SNAP_REACH_S]
        if near:
            setattr(c, attr, min(near, key=lambda x: abs(x - t)))
    if style.snap_sentences and not c.whole:  # a dialogue stretch ends where the reader said
        _snap_sentences(c, segments, duration, ix)
    words = ix.words
    if ix.starts_sorted and ix.ends_sorted:
        # Words are chronological: the first word the edge is inside of is the one to move
        # past (the loops below end after one word unless words overlap).
        nxt = 0
        while True:
            i = max(nxt, bisect_right(ix.ends, c.start))
            if i >= bisect_left(ix.starts, c.start):
                break
            w = words[i]
            c.start = max(0.0, w["start"] - min(0.05, ix.gap_before(w["start"]) / 2))
            nxt = i + 1
        bound = len(words)
        while True:
            j = min(bisect_left(ix.starts, c.end), bound) - 1
            if j < bisect_right(ix.ends, c.end):
                break
            c.end = min(duration, words[j]["end"] + min(0.15, ix.gap_after(words[j]["end"]) / 2))
            bound = j
    else:
        for w in words:  # words are chronological, so each fix can only move outward
            if w["start"] < c.start < w["end"]:
                c.start = max(0.0, w["start"] - min(0.05, ix.gap_before(w["start"]) / 2))
        for w in reversed(words):
            if w["start"] < c.end < w["end"]:
                c.end = min(duration, w["end"] + min(0.15, ix.gap_after(w["end"]) / 2))
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


def label_for(signals: Mapping[str, Any], style: Style, hint: str | None = None) -> str:
    """The label of a moment: its keyword ``hint`` as a slug, else "laugh" or "loud-moment"
    from its ``signals``, else "<style>-moment"."""
    if hint:
        return slugify(hint, 30)
    if signals.get("laughter", 0) > 0.5:
        return "laugh"
    if signals.get("loudness", 0) > 0.6:
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
    edge_cuts: scene_mod.EdgeCuts | None = None  # the cuts are detected around candidate edges only


def text_signals(ps: PartSignals, style: Style, name: str) -> None:
    """Fill the style-dependent transcript signals of ``ps``: the keyword and laughter hits
    and the ``speech`` and ``laughter`` curves. The keyword weights are, in this order, damped
    by how often the part says each phrase (styles with ``repeat_damping``,
    :func:`damp_keywords`), cut to :data:`DELIVERY_FLAT` for a hype word said flat (styles with
    ``delivery_cue``, :func:`gate_delivery`), and profanity is gated against the loudness
    (styles with ``profanity_alone`` False, :func:`gate_profanity`, which then sees the
    weights left). Laughter is never damped or gated. ``name``: the source's, for the log."""
    n, sig, words = ps.n, ps.sig, ps.words
    kw_hits = keyword_hits(words, style.keywords)
    laugh_hits = laughter_hits(words, flat_markers(ps.segments))
    if style.repeat_damping:
        kw_hits = damp_keywords(kw_hits, n * DT)
    if style.delivery_cue:
        kw_hits = gate_delivery(kw_hits, laugh_hits, sig["loudness"])
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
    style weighs it (:func:`~cutpoint.scenes.scores_scenes`), else zeros (it would add nothing to
    the score). Cuts found only around edges (``edge_cuts``) are replaced by a full pass first."""
    if not ps.scenes:
        return
    if not scene_mod.scores_scenes(style):
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
        "source": str(src),
        "offset": round(offset, 3),
        "expected_audio_s": round(expected, 3),
        "audio_s": round(got, 3),
        "missing_s": round(max(0.0, short), 3),
        "failure_time": round(offset + got, 3),
        "messages": messages,
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


def find_resume_point(src: Path, fail_t: float, duration: float) -> tuple[float, float] | None:
    """First readable time after ``fail_t``, found with a few small probe reads.

    Probes at growing distances (5, 10, 20, ... s) and stops at the first good
    read, then bisects ``RESUME_REFINE`` times toward the failure to lose less
    footage. Each probe decodes only ~3 s (:data:`READ_PROBE_S`), and a failed spot is
    never retried.

    Returns:
        Absolute resume time and where the unreadable stretch before it ends (the last
        failed probe, or ``fail_t``, plus :data:`READ_PROBE_S`, at most the first good probe:
        the scene pass of the recovered part never seeks before it,
        :func:`~cutpoint.scenes._lead_in`), or None if nothing readable was found.
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
    return round(min(good + RESUME_MARGIN_S, duration), 3), round(
        min(last_bad + READ_PROBE_S, good), 3
    )


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
    read_checked: Callable[[MediaInfo, dict[str, Any] | None], None] | None = None,
    text: bool = True,
    on_transcript: Callable[[], None] | None = None,
    cpu_busy: Callable[[], bool] | None = None,
    readable_from: float | None = None,
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
    before loudness, Whisper and the scene cuts.
    ``on_transcript`` is called once the transcript is there (made or cached), before the
    scene cuts. ``cpu_busy`` says whether a CPU-heavy step of the caller (the background HUD
    OCR) runs right now (None: none does); the scene cuts split their decode between NVDEC and
    the CPU only when it does not, nor Whisper on the CPU (:func:`~cutpoint.scenes._hybrid_plan`).
    ``readable_from`` (a recovered part): where the unreadable stretch before ``offset`` ends,
    for :func:`~cutpoint.scenes.scene_cuts`.
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
    overlap: scene_mod._SceneJob | None = None  # the scene cuts, run alongside Whisper
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
        _audio_clock(src, info, offset, cdir)
        made = not wav.exists() or cached_transcript(tr, cdir) is None
        if not wav.exists():
            with _watch_extract(wav, span, extracted if report else None):
                res = extract_audio(io, wav, info, start=offset, tracks=tracks)
            _write_clock(cdir)
        tick("audio_extract_s", t0)
        read_error = check_read(src, info, offset, wav, cdir, res)
        if read_error:
            warn_read_error(src, read_error, recover)
            span = read_error["audio_s"]  # analyse only what was actually read
        if read_checked is not None:
            read_checked(info, read_error)

        t0 = time.perf_counter()
        lufs_data = _loudness(wav, cdir)
        tick("loudness_s", t0)

        t0 = time.perf_counter()
        try:
            if (
                use_scenes
                and not scene_mod.edge_windows(style, span)
                and scene_mod.scene_overlap()
                and cache.load_json(cdir / scene_mod.SCENES_FILE) is None
                and cached_transcript(tr, cdir) is None
            ):
                tr.load()  # the memory guard runs before the scene thread takes any VRAM
                # Whisper runs meanwhile: the split decode only while it is on the GPU
                free = getattr(tr, "device", None) == "cuda" and not (cpu_busy and cpu_busy())
                limit = span if read_error else None
                lead = {"readable_from": readable_from} if readable_from is not None else {}
                overlap = scene_mod._SceneJob(io, cdir, span, offset, limit, free, **lead)
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
            read_checked(info, None)

    cuts: list[float] = []
    edge_cuts: scene_mod.EdgeCuts | None = None
    # A style that does not weigh the scene curve: cuts only around candidate edges (#730), in a
    # container that seeks exactly (MPEG-TS takes the full pass, #976)
    windowed = use_scenes and scene_mod.edge_windows(style, span) and scene_mod._seeks_exactly(io)
    if windowed:  # a full pass already cached is used as it is
        if (found := cache.load_json(cdir / scene_mod.SCENES_FILE)) is not None:
            cuts = found
        else:
            stages["scene_windows"] = True
            edge_cuts = scene_mod.EdgeCuts(
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
            opts: dict[str, Any] = {} if cpu_busy is not None and cpu_busy() else {"cpu_free": True}
            if readable_from is not None:
                opts["readable_from"] = readable_from
            cuts = scene_mod.scene_cuts(
                io, cdir, span, offset, limit=span if read_error else None, **opts
            )
            took = time.perf_counter() - t0
        add("scenes_s", took)
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
        segments=shift_segments(segments, offset),
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
    ref: float | None = None,
) -> list[Candidate]:
    """Candidates of one analysed part (:func:`analyze_source`, or a :class:`GridPart`), in
    absolute source time. ``ref``: the source's :func:`pick_reference` (None: the part's own,
    for a source read in one part)."""
    src, span, offset = res.source, res.duration, res.offset
    ps = res.signals
    if ps is None:
        raise ValueError(f"{src}: no signals to pick candidates from")
    n, sig = ps.n, ps.sig
    segments = ps.segments
    avail = scorer.available_for()
    if cam_ctx is not None:
        cands, avail = _action_cam_candidates(
            src, n, span, sig, offset, style, cam_ctx, max_candidates, segments, timings, avail
        )
    else:
        usual = usual_count(style) if style.open_list and max_candidates is None else None
        # Picked on the raw weighted sum: the cut-offs do not depend on what is available.
        combined = scorer.combined(sig, style)
        cap = candidate_cap(style, max_candidates)
        raw = _pick_windows(src, combined, sig, style, span, cap, usual, ref)
        # The first ``usual`` picks are finished as they always were; the picks past them only
        # fill what those leave free, so a longer list never changes the top of a shorter one.
        head, tail = (raw[:usual], raw[usual:]) if usual is not None else (raw, [])
        cands = _merge(head, style)
        _cover_edges(ps, cands)
        snap_index = _SnapIndex(segments)
        for c in cands:
            _snap(c, ps.cuts, segments, span, style, snap_index)
        cands = _resolve_overlaps(cands, style)
        taken = [(c.start, c.end) for c in cands]
        tail = [c for c in tail if not any(a <= c.peak_t <= b for a, b in taken)]
        cands += _place_free(tail, taken, ps, span, style)
    res.avail = avail
    return describe_candidates(cands, style, ps, offset, avail)


def pick_sources(
    parts: Sequence[SourceResult],
    style: Style,
    max_candidates: int | None,
    timings: dict[str, float],
) -> None:
    """:func:`pick_candidates` for each of ``parts`` (not action-cam); the recovered parts of
    one source share its :func:`pick_reference`."""
    by_source: dict[Path, list[SourceResult]] = {}
    for r in parts:
        by_source.setdefault(r.source, []).append(r)
    for rs in by_source.values():
        ref = pick_reference(rs, style)
        for r in rs:
            r.candidates = pick_candidates(r, style, max_candidates, timings, ref=ref)


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
    # a label names what was said by the style's own weight, before repeat damping and the
    # delivery cue (#1071): those decide the score, not what the moment is about
    said_w = {k.rstrip("*"): w for k, w in style.keywords.items()}
    if avail is None:
        avail = scorer.available_for()
    for c in cands:
        c.score = scorer.score(c.peak, style, avail, c.must_include)
        inside = [(t, w, p) for t, w, p in kw_hits + laugh_hits if c.start <= t <= c.end]
        c.keywords = [p for _, _, p in sorted(inside, key=lambda h: -h[1])]
        c.keywords = list(dict.fromkeys(c.keywords))[:5]
        named = sorted(((said_w.get(p, w), p) for _, w, p in inside), key=lambda h: -h[0])
        strong = [p for w, p in named if w >= 0.5 and p not in PROFANE]  # a bare swear: no label
        if not c.label_hint:  # action-cam picks bring their own ("jump 0.6s", "crash 7g")
            c.label_hint = strong[0] if strong else ""
        said = [w for w in words if c.start <= w["start"] <= c.end]
        c.transcript, c.transcript_unsure = _transcript_snippet(said)
        if offset:  # back to absolute source time: clips cut the original file
            c.start, c.end = round(c.start + offset, 3), round(c.end + offset, 3)
            c.peak_t += offset
    return cands


def _fit_free(c: Candidate, taken: list[tuple[float, float]], span: float, style: Style) -> bool:
    """Keep ``c`` out of the ``taken`` windows (existing candidates), at least ``min_len``
    long: moved within the free stretch around its event; False if that is too short."""
    lo, hi = intervals.free_around(c.peak_t, taken, 0.0, span)
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
            stretches.setdefault(intervals.free_around(c.peak_t, taken, 0.0, span), []).append(c)
    cands = [m for group in stretches.values() for m in _merge(group, style)]
    _cover_edges(ps, cands)
    snap_index = _SnapIndex(ps.segments)
    for c in cands:
        _snap(c, ps.cuts, ps.segments, span, style, snap_index)
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
                label_hint=title_slug(s.text, default="") or slugify(s.name, 30),
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


@dataclass(frozen=True)
class CachedSource:
    """What the caches of an analysis hold for one of its sources (:func:`cached_sources`): its
    ``index`` in highlights.json's ``sources``, its cache folder ``cdir`` (None while ``gap`` is
    ``recovered`` or ``offline``), ``span`` (0: none recorded), which of the cached loudness,
    transcript and full scene pass (:data:`~cutpoint.scenes.SCENES_FILE`) are there, and whether its
    scene cuts come from edge windows instead (``windowed``, :func:`~cutpoint.scenes.edge_windows`).
    ``gap``: why it gets no grid, None when it gets one: ``recovered`` (read in parts), ``offline``
    (not reachable), ``length`` (no length recorded) or ``caches`` (nothing cached to build
    from)."""

    index: int
    source: Path
    gap: str | None
    cdir: Path | None = None
    span: float = 0.0
    loudness: bool = False
    transcript: bool = False
    scenes: bool = False
    windowed: bool = False


def cached_sources(hl: dict[str, Any], style: Style, *, create: bool = True) -> list[CachedSource]:
    """What the caches hold for each source of the analysis ``hl`` with ``style``, and why a
    source gets no grid of :func:`cached_grid_parts` (``CachedSource.gap``). Only looks for
    the cache files: nothing is read or decoded. ``create``: make a missing cache folder (as
    :func:`cache.source_dir`; False leaves none behind, ``cutpoint rerank``'s readiness)."""
    scenes_on = bool((hl.get("settings") or {}).get("scenes", True))
    entries = hl.get("sources") or []
    recovered = {str(e.get("path")) for e in entries if e.get("offset") or e.get("read_errors")}
    out: list[CachedSource] = []
    for i, e in enumerate(entries):
        src = Path(str(e.get("path")))  # none recorded: a path no cache folder has (offline)
        if str(e.get("path")) in recovered:
            out.append(CachedSource(i, src, "recovered"))
            continue
        try:
            cdir = cache.source_dir(src, create=create)
        except OSError:
            out.append(CachedSource(i, src, "offline"))
            continue
        loud = (cdir / "loudness.json").is_file()
        heard = cdir.is_dir() and any(cdir.glob("transcript-*.json"))
        found = (cdir / scene_mod.SCENES_FILE).is_file()
        span = float(e.get("analyzed_s") or e.get("duration") or 0.0)
        windowed = bool(span) and scenes_on and not found and scene_mod.edge_windows(style, span)
        if not span:
            gap: str | None = "length"  # no source length recorded (hand-made file)
        elif not (loud or heard or found or windowed):
            gap = "caches"  # nothing cached to build from
        else:
            gap = None
        out.append(CachedSource(i, src, gap, cdir, span, loud, heard, found, windowed))
    return out


def cached_grid_parts(hl: dict[str, Any], style: Style) -> list[GridPart]:
    """The grids of the analysis ``hl`` as its caches give them (what :func:`analyze_source` built:
    the loudness, the transcript of the analysis's model and language, the scene cuts). A source
    :func:`cached_sources` gives a ``gap`` has none: one with a recovered part, offline, without a
    recorded length, or whose caches are gone, unless the style detects its scene cuts around
    candidate edges (:func:`~cutpoint.scenes.edge_windows`): such a source (also one without audio,
    so without loudness or transcript) gets its part, the cuts decoded when an edge needs them. Nor
    has one whose cache files cannot be read. Otherwise a source without
    :data:`~cutpoint.scenes.SCENES_FILE` gets no scene curve or cuts (with a style that weighs the
    curve, an INFO line names it; ``analyze`` detects them again)."""
    settings = hl.get("settings") or {}
    scenes = bool(settings.get("scenes", True))
    out: list[GridPart] = []
    for c in cached_sources(hl, style):
        if c.gap is not None or c.cdir is None:
            continue
        src, cdir, span = c.source, c.cdir, c.span
        loud = cache.load_json(cdir / "loudness.json")
        segments = source_segments(src, settings.get("language"), settings.get("whisper_model"))
        found = cache.load_json(cdir / scene_mod.SCENES_FILE)
        # EdgeCuts serve it
        windowed = scenes and found is None and scene_mod.edge_windows(style, span)
        if loud is None and segments is None and found is None and not windowed:
            continue  # the cache files there cannot be read
        if scenes and found is None and scene_mod.scores_scenes(style):
            logger.info(
                "%s: no scene cuts cached; its scene score is missing until analyze runs again",
                src.name,
            )
        cuts = (found or []) if scenes else None
        lufs = (np.asarray(loud["t"]) - 0.2, np.asarray(loud["m"])) if loud else None
        ps = part_signals(span, lufs, segments or [], cuts, style, src.name)
        if windowed:  # as the analysis did: cuts around edges (scenes-windows.json, else decoded)
            ps.edge_cuts = scene_mod.EdgeCuts(remap_source(src), cdir, span)
        out.append(GridPart(src, c.index, 0.0, span, ps))
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
    snap_index = _SnapIndex(segments)
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
        _snap(c, [], segments, span, style, snap_index)
        cands.append(c)
    timings["action_cam_s"] = round(timings.get("action_cam_s", 0.0) + time.perf_counter() - t0, 2)
    return cands, src_avail


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
    "visual_light": "keyframes",  # without a CUDA GPU: the AI Chat's light frame index (#989)
    "ocr": "keyframes",
    # the on-screen overlays of non-gameplay footage (#1069 (c), events.overlays)
    "overlay": "keyframes",
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
    # why the keyframes step runs without the visual layer the crashes step reads (the
    # light frame index in its place: :data:`SKIP_NO_CUDA`)
    no_visual: str | None = None

    def all_layers(self) -> tuple[str, ...]:
        """Every planned layer, in the order :func:`run_events` runs them."""
        return tuple(x for x in LAYER_STEP if any(x in v for v in self.layers.values()))


# The --json warning of action-cam sources without camera telemetry (#939).
NO_TELEMETRY_WARNING = "action_cam_no_telemetry"


def _no_telemetry_fallback(
    style: Style, sources: Sequence[Path], unreadable: bool = False
) -> tuple[Style, dict[str, str]]:
    """The ``action`` style an action-cam ``style`` falls back to when no source has camera
    telemetry, or (``unreadable``) the telemetry of every source failed to read (with the
    ``--must-include-crash`` of ``style``), and the ``settings.style_fallback`` record; logs a
    WARNING and the ``--json`` warning :data:`NO_TELEMETRY_WARNING` naming every source."""
    from cutpoint.styles import get_style  # noqa: PLC0415

    used = replace(get_style("action"), must_include_crash=style.must_include_crash)
    named = "action-cam" if style.name == ACTION_CAM else f"{style.name} (an action-cam style)"
    if unreadable:
        names = ", ".join(s.name for s in sources)
        message = (
            f"No camera telemetry could be read from any source ({names}); {named} needs "
            f"it, so the {used.name} style was used for this run (nothing was cached: a later "
            f"run reads it again). Give --style {used.name} (or auto) to choose it"
        )
    else:
        message = (
            f"No camera telemetry in any source; {named} needs GoPro or Insta360 X3 telemetry, "
            f"so the {used.name} style was used. Give --style {used.name} (or auto) to choose it"
        )
    logger.warning("%s", message)
    jsonout.warn(NO_TELEMETRY_WARNING, message, sources)
    reason = "telemetry not read" if unreadable else "no telemetry"
    return used, {"from": style.name, "reason": reason}


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


def _light_index(cat: str, visual_index: bool) -> bool:
    """Whether a run without a CUDA GPU writes the light frame index for the AI Chat (#989),
    for footage of category ``cat``: only where a GPU run writes the full index, action-cam,
    action (the ``sport`` style too: :func:`layer_category`) and general footage, and the
    others with ``--visual-index`` (``visual_index`` in config.json)."""
    return cat in (category.ACTION_CAM, category.ACTION, category.GENERAL) or visual_index


def plan_layers(
    style: Style,
    sources: Sequence[Path],
    skip: Collection[str],
    cat: str | None = None,
    visual_index: bool = False,
    cuda: Callable[[], bool] | None = None,
    quiet: bool = False,
) -> LayerPlan:
    """Which event layers the reactions, keyframes and crashes steps run.

    ``cat``: the footage category (:mod:`cutpoint.category`; None: :func:`layer_category`
    of ``style``). Reactions: the PANNs audio events (plus the voice layer for action-cam
    footage, whose crash audio score reads it). Keyframes: HUD OCR for gameplay, the CLIP
    visual layer for action-cam, action and general footage (only with a CUDA GPU), and for
    other footage the overlay layer when the style looks for overlays
    (:func:`~cutpoint.events.layers.overlay_kinds`: a scoreboard, gauges, lower-thirds; it reads
    them only when a vision model answers); ``visual_index``
    (``--visual-index``) adds the CLIP visual layer to the keyframes of gameplay (before the
    HUD text) and talk too, for the assistant's visual search only. ``cuda``: whether a CUDA
    GPU is there (None: torch's own check, :func:`_cuda`; ``cutpoint ui`` passes a child
    process's answer, so it never imports torch). Without a CUDA GPU the light frame index for
    the AI Chat (``visual_light``, CLIP on the CPU, chat only) takes the CLIP visual layer's
    place where :func:`_light_index` says so; ``no_visual`` then tells the crashes step why it
    has no visual layer. Crashes:
    action-cam, action and general footage, or any source with telemetry. A step whose events extra
    or model files are missing is left out with one INFO line naming the fix (DEBUG with ``quiet``);
    nothing is downloaded.
    """
    log = logger.debug if quiet else logger.info
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
        # the overlays the style looks for (none for gameplay: its HUD is the OCR layer's)
        overlay = ("overlay",) if not plan.game and overlay_kinds(style.name, style.base) else ()
        if plan.game:
            want["keyframes"] = ("ocr", "visual") if visual_index else ("ocr",)
        elif action or visual_index:
            want["keyframes"] = ("visual", *overlay)
        elif overlay:
            want["keyframes"] = overlay
        else:
            plan.skipped["keyframes"] = f"no keyframe layer for the {style.name} style"
    if not want:
        return plan
    missing = [m for m in EVENTS_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        from cutpoint.extras import LITE, install_hint  # noqa: PLC0415

        names = [LABELS[s].lower() for s in want]
        log(
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
    light = plan.skipped.get("keyframes") != SKIP_OPTION and _light_index(cat, visual_index)
    if ("visual" in want.get("keyframes", ()) or light) and not (cuda or _cuda)():
        # gameplay: the HUD text stays
        rest = tuple(x for x in want.get("keyframes", ()) if x != "visual")
        if light:  # run_events logs its INFO line with the estimate
            want["keyframes"] = (*rest, "visual_light")
            plan.skipped.pop("keyframes", None)  # a style without a keyframe layer of its own
            plan.no_visual = SKIP_NO_CUDA  # crashes.json still has no visual layer
        else:
            log(
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
        # reactions without the voice layer, keyframes of gameplay without the visual index,
        # keyframes of other footage with the overlays alone
        visual = plan.game or "overlay" in layers
        rest = tuple(
            x for x in layers if x not in ("voice", *(("visual", "visual_light") if visual else ()))
        )
        if rest != layers and rest and not event_paths.missing_files(rest):
            want[step] = rest
            continue
        need = " ".join(dict.fromkeys(comp for comp, _ in files))
        plan.skipped[step] = f"model files missing (cutpoint setup --only {need})"
        del want[step]
    if components:
        log(
            "Model files are missing for some analysis steps (nothing is downloaded now); "
            "to add them: cutpoint setup --only %s",
            " ".join(sorted(components, key=COMPONENTS.index)),
        )
    return plan


REASON_CHARS = 120  # the reason of a failed OCR layer is cut to this


def _ocr_failed(err: BaseException) -> tuple[str, list[str]]:
    """The WARNING (and ``step_failed`` message) of an OCR layer that failed with ``err``,
    naming what it costs, and the paths of the sources it did not read (none when the
    failure is not tagged with them, :func:`~cutpoint.events.pipeline.ocr_failure`). An
    out-of-memory error never gets here: it stops the run (:class:`_StepHook`)."""
    from cutpoint.events.pipeline import ocr_failure  # noqa: PLC0415

    lines = str(err).strip().splitlines()
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
    elif where.fit:  # after discovery: the presets are not checked (no "no preset" claim)
        message = f"OCR failed in the HUD fit check ({reason}): {effect} every source"
    elif where.install:  # after discovery: the preset was not written (no "no preset" claim)
        message = f"OCR failed writing the HUD preset ({reason}): {effect} every source"
    elif not paths:
        message = f"OCR failed ({reason}): {effect} every source"
    else:
        later = len(paths) - 1
        rest = f" and {later} later source{'s' if later != 1 else ''}" if later else ""
        message = f"OCR failed ({reason}): {effect} {Path(paths[0]).name}{rest}"
    return message, paths


class _StepHook:
    """``run_events``'s ``stage`` for :func:`analyze`: each layer reports as its step; a
    layer that fails fails its step (an ERROR line and a ``step_failed`` warning; for the
    OCR layer a WARNING naming the sources without menu filtering and HUD events, see
    :func:`_ocr_failed`), and the next steps still run. A layer in ``optional`` that fails
    only gets a WARNING and a ``step_failed`` warning: its step is not failed. Out of memory
    in the OCR layer stops the run instead, as ``low_memory`` (resumable; the layer's reads
    and discovery raise it themselves, this catches the rest)."""

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
            raise  # (visual_light never gets here: its own WARNING, pipeline._light_layer)
        except Exception as err:
            logger.debug("%s layer failed", layer, exc_info=True)
            files: list[str] = []
            if layer == "ocr":  # analyze goes on, ranked without the HUD: a WARNING
                memguard.raise_if_out_of_memory(err, "the HUD OCR", "the sources not read yet")
                message, files = _ocr_failed(err)
                logger.warning("%s", message)
            elif layer == "visual_light":  # chat only: the analysis does not need it
                message = f"Frame index for the AI Chat failed: {err}; continuing without it"
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
    context: str | None = None,
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """The dialogue step as a function of the audio events (events.json entries with source,
    ``start``, ``label`` and ``confidence``): reads the ``transcripts`` with ``backend`` and
    returns the dialogue events. Records ``timings["dialogue_s"]`` and appends to ``ran`` when
    called. ``context``: the user's ``--context`` words, given to the reader's prompt."""
    from cutpoint.events import dialogue  # noqa: PLC0415

    def read(audio: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ran.append(True)
        t0 = time.perf_counter()
        try:
            found = dialogue.read_all(
                transcripts,
                audio,
                backend,
                lambda pct, name: steps.update(pct, name),
                context=context,
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
    context: str | None = None,
    vision: bool | None = None,
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
    ``--ocr-det-skip`` (``OcrParams.det_skip``); ``context``: ``--context``, for the crash
    layer's activity profile; ``vision``: ``--vision`` / ``--no-vision`` (False: no HUD map,
    top-up or overlays, :func:`~cutpoint.events.pipeline.vision_off`)."""
    from cutpoint.crash import no_visual_reason  # noqa: PLC0415
    from cutpoint.events.audio_events import AudioParams  # noqa: PLC0415
    from cutpoint.events.common import pending_highlights  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415
    from cutpoint.events.pipeline import run_events  # noqa: PLC0415
    from cutpoint.events.vlm_windows import VlmParams  # noqa: PLC0415

    tags, game = _ocr_options(project)
    why = plan.skipped.get("keyframes") or plan.no_visual
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
                stage=_StepHook(steps, {"visual_light", *(("visual",) if plan.game else ())}),
                cached_ocr=plan.game,
                llm=llm,
                prefetch=prefetch,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                discover_hint=DISCOVER_HINT,
                batch_size=batch_size,
                dialogue=dialogue,
                context=context,
                vision=vision,
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
        "label": label_for(c.signals, style, c.label_hint),
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
    reranked: dict[str, Any] | None = None,
    rerank_backup: str | None = None,
    ids: Mapping[tuple[str, float, float], str] | None = None,
    last_candidate_id: int = 0,
    context: str | None = None,
    context_recorded: bool = False,
    vision_detect_all: bool = False,
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
            below its cut (:func:`~cutpoint.vision_step.vision_band`) are cut to this many, closest
            to the cut first (None = the whole band).
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
            (:func:`~cutpoint.reanalyze.reanalyze`): kept as ``total_s``, and this run's own time is
            recorded as ``timings.rerank_s``. None: this run's time is ``total_s``.
        detection: The ``detection`` block of the auto analysis this run re-ranks
            (:func:`~cutpoint.reanalyze.reanalyze`, with the style it chose): recorded again as it
            is, and its category plans the layers. Nothing is detected again.
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
        crash_moments: Make the crashes of crashes.json moments
            (:func:`~cutpoint.crash_moments.add_crash_candidates`, every style but action-cam);
            False (``360 analyze``) leaves them out and records
            ``settings.crash_moments`` false, which :func:`~cutpoint.reanalyze.reanalyze` keeps.
        footage: The caller's :class:`~cutpoint.category.FootageVision` for ``--style auto``
            (shared with its own detection; not closed here). None: one of this run.
        vision_time_share: The vision step's time cap, a share of the analysis before it
            (:func:`~cutpoint.vision_step.vision_budget`); None = ``vision_time_share`` in
            config.json.
        telemetry_fallback: An action-cam ``style`` (built-in or a style file over it) whose
            sources all lack camera telemetry (:func:`has_telemetry`) runs as ``action``
            instead, recorded as ``settings.style_requested`` and ``settings.style_fallback``
            with an ``action_cam_no_telemetry`` warning (#939). False (``360 analyze``): the
            style runs as given.
        reranked: The ``settings.reranked`` record of a project re-ranked to another style
            (:mod:`cutpoint.rerank`, :func:`~cutpoint.reanalyze.reanalyze`): recorded again, with
            ``settings.style_requested`` the style's name, the user's choice. ``detection``
            is then the earlier auto analysis's block, kept as it is but not acted on: the
            layers follow ``style``.
        rerank_backup: ``settings.rerank_backup`` (:func:`~cutpoint.reanalyze.backup_digest` of
            the backup a re-rank kept, :func:`~cutpoint.reanalyze.reanalyze`): recorded again, so
            the backup
            still gives ids back. None: not recorded, unless the earlier ``out_dir/highlights.json``
            is linked to the backup file next to it
            (:func:`~cutpoint.reanalyze.project_backup_link`, #1009): that link is carried on.
            Otherwise a fresh analysis is not linked to a backup.
        ids: For a re-analysis (:func:`~cutpoint.reanalyze.reanalyze`, ``cutpoint rerank
            --analyze``): the ids of the earlier candidates by
            :func:`~cutpoint.reanalyze.span_key`
            (:func:`~cutpoint.reanalyze.earlier_ids`). A candidate with the span of one keeps its
            id, the others get the next ids after ``last_candidate_id``. None: the ids of an earlier
            ``out_dir/highlights.json`` that is an analysis are taken the same way
            (:func:`~cutpoint.reanalyze.project_ids`, #1009), so a fresh run into a project
            (``analyze``, ``edit``, ``360 analyze``) keeps the ids of the spans it finds again;
            without one every candidate is numbered from ``h001``.
        last_candidate_id: The highest id the earlier analysis gave (with ``ids`` only).
        context: ``--context``, the user's words about the footage (#1068), recorded as
            ``settings.context``: a hint in the footage question of ``--style auto`` (not
            with ``footage``, which brings its own), it picks the crash step's activity
            profile as ``crashes --context`` does, and it goes into the vision rating and
            dialogue prompts. No scoring weight reads it. None or blank: as without it.
        context_recorded: ``context`` is the recorded ``settings.context`` of a re-analysis
            (:func:`~cutpoint.reanalyze.reanalyze`), not words the user gave now: the crash step
            keeps the project's activity.json then, as ``crashes`` without ``--context`` does.
        vision_detect_all: ``--vision-detect-all`` (#1069; the caller has turned it off for
            ``--no-vision``): ``--style auto`` shows every source to the vision model; it only
            settles sources (:func:`cutpoint.category.decide`), recorded as
            ``settings.vision_detect_all`` (only when on) for a re-analysis and ``cutpoint
            ui``. With ``footage``, its own ``every_source`` decides.

    Returns:
        Path of the written highlights.json.
    """
    if ids is None:
        ids, last_candidate_id = reanalyze.project_ids(out_dir)
        if rerank_backup is None:
            rerank_backup = reanalyze.project_backup_link(out_dir)
    with progress.run(ANALYZE_STEPS, use=layers) as steps:
        opts = AnalyzeOptions(
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
            reranked=reranked,
            rerank_backup=rerank_backup,
            ids=ids,
            last_candidate_id=last_candidate_id,
            context=footage_context(context),
            context_recorded=context_recorded,
            vision_detect_all=footage.every_source if footage is not None else vision_detect_all,
        )
        return _analyze(target, style, out_dir, steps, opts=opts)


@dataclass(frozen=True)
class AnalyzeOptions:
    """The keyword options of :func:`analyze` (see there) as :func:`_analyze` and its steps read
    them: ``visual_index`` and ``ocr_det_skip`` as bools, ``context`` through
    :func:`~cutpoint.utils.footage_context`, ``vision_detect_all`` ``footage``'s own
    ``every_source`` when it is given, and ``ids``, ``last_candidate_id`` and ``rerank_backup``
    from the project when :func:`analyze` got no ``ids``. :func:`_analyze` replaces
    ``vision_detect_all`` (off with ``--no-vision``) and ``use_scenes`` (and the style's own)
    before the steps read them."""

    vision: bool | None
    model: str | None
    language: str | None
    use_scenes: bool
    max_candidates: int | None
    vision_top: int | None
    ollama_url: str | None
    ollama_model: str | None
    exclude: Sequence[str]
    recursive: bool
    recover: bool
    layers: bool
    skip: Collection[str]
    earlier_total_s: float | None
    detection: dict[str, Any] | None
    batch_size: int
    visual_index: bool
    ocr_det_skip: bool
    crash_moments: bool
    footage: category.FootageVision | None
    vision_time_share: float | None
    telemetry_fallback: bool
    reranked: dict[str, Any] | None
    rerank_backup: str | None
    ids: Mapping[tuple[str, float, float], str] | None
    last_candidate_id: int
    context: str | None
    context_recorded: bool
    vision_detect_all: bool


@dataclass
class _StylePlan:
    """The style an analysis starts its transcription with (:func:`_start_style`)."""

    style: Style
    cat: str | None  # the category whose layers run (None: the style's own)
    probes: list[category.SourceProbe]  # --style auto: decided again with the speech coverage
    pending: bool  # --style auto: talk is told from general once the speech coverage is known
    fallback: str | None  # --style auto: category.choose_style's fallback
    style_fallback: dict[str, str] | None  # #939: an action-cam style without telemetry


@dataclass
class _ActionCam:
    """The action-cam context of an analysis (:func:`_prepare_action_cam`)."""

    ctx: Any  # cutpoint.actioncam.ActionCamContext
    durations: dict[str, float]  # per source, for actioncam.prepare again after the events
    telemetry_s: float
    unreadable: bool = False  # every source's telemetry failed to read (see the fallback)


def _analyze(
    target: Path | Sequence[Path],
    style: Style | AutoRequest,
    out_dir: Path,
    steps: progress.Run,
    *,
    opts: AnalyzeOptions,
) -> Path:
    """:func:`analyze`, reporting to ``steps``."""
    t_start = time.perf_counter()
    steps.start("transcribe")
    sources = find_sources(target, opts.exclude, opts.recursive)
    # joined MPEG-TS/PS recordings whose timestamps jump are read through a stream copy (#1052)
    timelines = tsrepair.prepare_sources(sources)
    # --style auto: probed before transcription, which needs the style's action-cam context, scene
    # setting and OCR prefetch; talk is told from general once the speech coverage is known.
    auto = style if isinstance(style, AutoRequest) else None
    if auto is not None and opts.vision is False and opts.footage is None:
        # --no-vision wins: nothing is shown to the model
        opts = replace(opts, vision_detect_all=False)
    start = _start_style(auto, style, sources, opts)
    style = start.style
    with (
        folders.output_dir(out_dir),  # a failed run leaves no empty project folder behind
        runstate.project_ready(out_dir),  # the run lock of a new project, now it exists
        cache.hold_outputs(out_dir, HELD_OUTPUTS) as held,
        # HUD OCR in the background while the GPU steps run; stopped before the folder goes
        _ocr_prefetch(
            style,
            sources,
            opts.skip,
            out_dir,
            opts.layers,
            start.cat,
            opts.batch_size,
            opts.ocr_det_skip,
        ) as prefetch,
    ):
        # ---- 1. transcribe: audio, loudness, Whisper and scene cuts per source
        scenes = opts.use_scenes
        opts = replace(opts, use_scenes=scenes and style.use_scenes)
        cam = _prepare_action_cam(style, sources, out_dir, _may_fall_back(auto, style, opts))
        if cam is not None and cam.unreadable:
            # every telemetry read failed: as footage without telemetry, for this run only
            style, style_fallback = _no_telemetry_fallback(style, sources, unreadable=True)
            start = replace(start, style=style, style_fallback=style_fallback)
            opts = replace(opts, use_scenes=scenes and style.use_scenes)
            cam = None
        tr = Transcriber(model=opts.model, language=opts.language)
        timings: dict[str, float] = {}
        if cam is not None:
            timings["telemetry_s"] = cam.telemetry_s
        results, transcript = _transcribe_sources(
            sources, style, tr, start.pending, cam, timings, prefetch, out_dir, steps, opts
        )
        style, cat, detection_rec = _final_style(auto, start, sources, results, timings, opts)
        steps.done()  # an error of the later steps never fails the transcription
        _summarize_read_errors(results, opts)
        hl = _highlights_record(
            target,
            style,
            detection_rec,
            start.style_fallback,
            tr,
            results,
            timelines,
            timings,
            opts,
        )

        # ---- 2-4. reactions, keyframes, crashes: the event layers (events.json)
        events_new = False
        if opts.layers:
            events_new = _run_event_steps(
                hl,
                results,
                sources,
                style,
                cat,
                transcript,
                timings,
                prefetch,
                out_dir,
                steps,
                opts,
            )

        # ---- 5. ranking: candidates rescored with the events
        ev = _rank_results(
            hl, results, sources, style, cam, events_new, timings, t_start, out_dir, steps, opts
        )
        selfcheck.apply(hl, ev)  # warns; changes no score (#1073)
        return _write_outputs(
            hl,
            ev,
            results,
            transcript,
            timings,
            t_start,
            opts.earlier_total_s,
            prefetch,
            held,
            out_dir,
        )


def _start_style(
    auto: AutoRequest | None,
    style: Style | AutoRequest,
    sources: Sequence[Path],
    opts: AnalyzeOptions,
) -> _StylePlan:
    """The style and layer category the transcription starts with: ``--style auto`` probes the
    sources (talk is told from general later, :func:`_final_style`), a re-run takes the recorded
    category, and an action-cam style without any telemetry runs as ``action``."""
    probes: list[category.SourceProbe] = []
    cat: str | None = None  # the category whose layers run (None: the style's own)
    pending = False
    fallback: str | None = None
    if auto is not None:
        probes, det = category.detect(
            sources,
            opts.vision,
            opts.ollama_url,
            opts.ollama_model,
            shared=opts.footage,
            context=opts.context,
            every_source=opts.vision_detect_all,
        )
        name, fallback = category.choose_style(det, auto.emphasis)
        if det.pending:
            logger.info("Footage looks %s so far (%s)", category.label(det.category), det.reason)
        style = auto.style(name)
        cat = det.category if fallback is None else None
        pending = det.pending  # the style's text signals wait for talk
    elif (
        opts.detection is not None and opts.reranked is None and not opts.detection.get("fallback")
    ):
        # a re-run: as decided
        stored = opts.detection.get("category")
        cat = stored if stored in category.CATEGORIES else None
    # action-cam without any telemetry: the action style, decided before the OCR prefetch, the
    # scene setting and the layer plan, which all follow the style
    style_fallback: dict[str, str] | None = None
    if _may_fall_back(auto, style, opts) and not any(has_telemetry(s) for s in sources):
        style, style_fallback = _no_telemetry_fallback(style, sources)
    return _StylePlan(style, cat, probes, pending, fallback, style_fallback)


def _may_fall_back(auto: AutoRequest | None, style: Style, opts: AnalyzeOptions) -> bool:
    """Whether an action-cam ``style`` runs as ``action`` when its footage has no usable
    telemetry (#939): given by name, not decided by a re-run, and not a 360 project."""
    return (
        auto is None
        and (opts.detection is None or opts.reranked is not None)
        and opts.telemetry_fallback
        and style.kind == ACTION_CAM
    )


def _prepare_action_cam(
    style: Style, sources: Sequence[Path], out_dir: Path, fall_back: bool = False
) -> _ActionCam | None:
    """An action-cam style's context (telemetry and features) before the transcription; None
    for any other style. ``fall_back``: when every source's telemetry fails to read, the
    context has ``unreadable`` set (and no telemetry.json) instead of raising the error."""
    if style.kind != ACTION_CAM:
        return None
    from cutpoint import actioncam  # noqa: PLC0415

    t0 = time.perf_counter()
    stale = out_dir / actioncam.FEATURES_DIR
    if stale.is_dir():
        for f in stale.glob("*.npz"):
            cache.remove(f)  # held: removed only when the run completes
    durations = {str(s): probe(remap_source(s)).duration for s in sources}
    cam_ctx = actioncam.prepare(sources, durations, out_dir, raise_if_all_failed=not fall_back)
    telemetry_s = round(time.perf_counter() - t0, 2)
    if fall_back and cam_ctx.unreadable and all(cam_ctx.telemetry[str(s)] is None for s in sources):
        return _ActionCam(cam_ctx, durations, telemetry_s, unreadable=True)
    _warn_missing_telemetry([s for s in sources if cam_ctx.telemetry[str(s)] is None])
    return _ActionCam(cam_ctx, durations, telemetry_s)


def _transcribe_sources(
    sources: Sequence[Path],
    style: Style,
    tr: Transcriber,
    pending: bool,
    cam: _ActionCam | None,
    timings: dict[str, float],
    prefetch: OcrPrefetch | None,
    out_dir: Path,
    steps: progress.Run,
    opts: AnalyzeOptions,
) -> tuple[list[SourceResult], str]:
    """The transcribe step's sources: audio, loudness, Whisper and scene cuts per source
    (:func:`analyze_source`), a part recovered after a read error a result of its own; Whisper
    is closed at the end.

    Returns:
        The results and the transcript.txt text.
    """

    def read_checked(src: Path, offset: float) -> Callable[..., None] | None:
        """``analyze_source``'s hook: offer the source to the OCR prefetch, which reads
        only a whole, clean source while Whisper runs on the GPU."""
        if prefetch is None:
            return None

        def offer(info: MediaInfo, rec: dict[str, Any] | None) -> None:
            ok = not offset and rec is None and tr.device == "cuda"
            prefetch.offer(src, round(info.duration, 3), ok)

        return offer

    def cpu_busy() -> bool:
        """The background HUD OCR (CPU-heavy) reads or is about to read a source: a source
        offered and not read yet (:attr:`~cutpoint.events.ocr_prefetch.OcrPrefetch.busy`)."""
        return prefetch is not None and prefetch.busy

    results: list[SourceResult] = []
    transcript_lines: list[str] = []
    progress.continuing("transcribe", _cached_transcripts(sources, tr), len(sources))
    for i, src in enumerate(sources):
        if opts.batch_size and i and not i % opts.batch_size:  # a batch boundary (--batch-size)
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

        offset, n_parts, first = 0.0, 0, len(results)
        readable_from: float | None = None  # where the damage before a recovered part ends
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
                opts.use_scenes,
                opts.max_candidates,
                timings,
                offset=offset,
                recover=opts.recover,
                # with --recover once every part is read: they share one reference
                pick=cam is None and not pending and not opts.recover,
                report=report if steps.active else None,
                read_checked=read_checked(src, offset),
                text=not pending,
                # progress.json from the first transcript on, before the scene cuts
                on_transcript=lambda: runstate.transcribed(out_dir, sources),
                cpu_busy=cpu_busy,
                readable_from=readable_from,
            )
            results.append(res)
            name = f"{src.name}@{offset:.0f}" if offset else src.name
            transcript_lines.append(f"# {src}" + (f" (from {clock(offset)})" if offset else ""))
            transcript_lines.extend(transcript_text(name, res.segments) or ["(no speech)"])
            transcript_lines.append("")
            rec = res.read_error
            if not (opts.recover and rec) or n_parts >= MAX_RECOVERED_PARTS:
                break
            cdir = cache.source_dir(src, offset)
            # a record older than unreadable_until: no lead-in for the next part's scene pass
            resume, until = rec.get("resumed_at"), rec.get("unreadable_until")
            if not resume:
                found = find_resume_point(remap_source(src), rec["failure_time"], res.info.duration)
                resume, until = found if found is not None else (None, None)
            if resume is None or resume <= offset:
                logger.warning(
                    "%s: no readable point found after %s; rest of file skipped",
                    src.name,
                    clock(rec["failure_time"]),
                )
                break
            rec["resumed_at"] = resume
            rec["unreadable_until"] = until
            rec["lost_s"] = round(resume - rec["failure_time"], 3)
            cache.save_json(cdir / READ_ERRORS_FILE, rec)
            logger.warning(
                "%s: resuming at %s (skipping %.1f s of unreadable footage)",
                src.name,
                clock(resume),
                rec["lost_s"],
            )
            offset, readable_from, n_parts = resume, until, n_parts + 1
        if opts.recover and cam is None and not pending:
            pick_sources(results[first:], style, opts.max_candidates, timings)
            n_cands = sum(len(r.candidates) for r in results[first:])
            logger.info("%s: %d candidates", src.name, n_cands)
        runstate.transcribed(out_dir, sources)  # a source without audio: once analyzed
    # Written with highlights.json at the end; the event layers read it from memory.
    transcript = "\n".join(transcript_lines)
    tr.close()  # Whisper out of memory before the next step loads a model
    free_models()
    _warn_transcript_fallback(tr)
    return results, transcript


def _final_style(
    auto: AutoRequest | None,
    start: _StylePlan,
    sources: Sequence[Path],
    results: list[SourceResult],
    timings: dict[str, float],
    opts: AnalyzeOptions,
) -> tuple[Style, str | None, dict[str, Any] | None]:
    """``--style auto``'s final decision, with the speech coverage: a pending style (talk or
    general) gets its transcript signals and candidates now.

    Returns:
        The style, the layer category and highlights.json's ``detection`` record.
    """
    style, cat, fallback = start.style, start.cat, start.fallback
    detection_rec = opts.detection  # highlights.json's record of an auto run (a re-run: as given)
    if auto is not None:  # the final decision, with the speech coverage
        det = category.decide(
            start.probes,
            category.source_speech(sources, [(r.source, r.duration, r.segments) for r in results]),
            every_source=opts.vision_detect_all and opts.vision is not False,
        )
        if start.pending:  # talk or general: this style's transcript signals and candidates
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
            read = [r for r in results if r.signals is not None]
            pick_sources(read, style, opts.max_candidates, timings)
        category.warn(det, style.name, fallback)
        detection_rec = det.record(auto.emphasis, style.name, fallback)
    return style, cat, detection_rec


def _summarize_read_errors(results: Sequence[SourceResult], opts: AnalyzeOptions) -> None:
    """One warning per source whose read stopped early."""
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
                if not opts.recover
                else " (could not recover)",
            )


def _highlights_record(
    target: Path | Sequence[Path],
    style: Style,
    detection_rec: dict[str, Any] | None,
    style_fallback: dict[str, str] | None,
    tr: Transcriber,
    results: list[SourceResult],
    timelines: Mapping[str, Any],
    timings: dict[str, float],
    opts: AnalyzeOptions,
) -> dict[str, Any]:
    """highlights.json's object before the event layers and the ranking fill it."""
    return {
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
            "scenes": opts.use_scenes,
            "vision_requested": bool(opts.vision),
            # --no-vision / "vision": false: no frames to a vision model (HUD map, overlays);
            # `events`, `rerank --analyze` and `resume` follow it
            **({"vision_off": True} if opts.vision is False else {}),
            "vision_used": False,
            "vision": None,
            "vision_top": opts.vision_top,
            # never its login: a replay takes it from config.json or OLLAMA_HOST (A639)
            "ollama_url": backends.without_login(opts.ollama_url)
            if opts.ollama_url
            else opts.ollama_url,
            "ollama_model": opts.ollama_model,
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
                str(Path(t).absolute()) for t in ([target] if isinstance(target, Path) else target)
            ],
            "exclude": list(opts.exclude),
            "recursive": opts.recursive,
            "language": opts.language,
            "max_candidates": opts.max_candidates,
            "recover": opts.recover,
            **({} if opts.crash_moments else {"crash_moments": False}),  # 360 analyze
            **(
                {"style_requested": AUTO}
                if detection_rec is not None and opts.reranked is None
                else {}
            ),
            # re-ranked to another style (cutpoint rerank): the user's choice
            **(
                {"style_requested": style.name, "reranked": opts.reranked}
                if opts.reranked is not None
                else {}
            ),
            # the backup of the last re-rank, which gives ids back (#1005)
            **({"rerank_backup": opts.rerank_backup} if opts.rerank_backup is not None else {}),
            # #939: an action-cam style without telemetry ran as `style` (action)
            **(
                {"style_requested": style_fallback["from"], "style_fallback": style_fallback}
                if style_fallback is not None
                else {}
            ),
            # --ocr-det-skip: `cutpoint events` reads HUD text as this run did (only when on)
            **({"ocr_det_skip": True} if opts.ocr_det_skip else {}),
            # --context (#1068): the crash step's profile, the vision and dialogue prompts;
            # `cutpoint events`, `rerank` and re-analyses read it from here (only when given)
            **({"context": opts.context} if opts.context else {}),
            # --vision-detect-all (#1069): how --style auto detected (only when on)
            **({"vision_detect_all": True} if opts.vision_detect_all and detection_rec else {}),
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
                **({"timeline": timelines[str(r.source)]} if str(r.source) in timelines else {}),
            }
            for r in results
        ],
        "candidates": [],
    }


def _run_event_steps(
    hl: dict[str, Any],
    results: list[SourceResult],
    sources: Sequence[Path],
    style: Style,
    cat: str | None,
    transcript: str,
    timings: dict[str, float],
    prefetch: OcrPrefetch | None,
    out_dir: Path,
    steps: progress.Run,
    opts: AnalyzeOptions,
) -> bool:
    """The event layers (reactions, keyframes, crashes: :func:`plan_layers`) and the dialogue
    step, with ``layers``.

    Returns:
        Whether the event layers wrote this run's events.json.
    """
    events_new = False
    plan = plan_layers(style, sources, opts.skip, cat, opts.visual_index)
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
    backend = _dialogue_backend(style, opts.skip, steps, spoken, opts.ollama_url, opts.ollama_model)
    ran: list[bool] = []
    reader = (
        _dialogue_reader(backend, spoken, steps, timings, ran, opts.context)
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
            ollama_url=opts.ollama_url,
            ollama_model=opts.ollama_model,
            batch_size=opts.batch_size,
            dialogue=(lambda layer1: reader(_audio_events(layer1))) if reader else None,
            det_skip=opts.ocr_det_skip,
            # a recorded context does not replace the project's activity.json (the
            # VLM layer still reads it from settings.context)
            context=None if opts.context_recorded else opts.context,
            vision=opts.vision,
        )
        timings["events_s"] = round(time.perf_counter() - t0, 2)
        free_models()
    if reader is not None and not ran:
        _dialogue_alone(out_dir, hl, transcript, steps, reader, [str(r.source) for r in results])
    return events_new


def _rank_results(
    hl: dict[str, Any],
    results: list[SourceResult],
    sources: Sequence[Path],
    style: Style,
    cam: _ActionCam | None,
    events_new: bool,
    timings: dict[str, float],
    t_start: float,
    out_dir: Path,
    steps: progress.Run,
    opts: AnalyzeOptions,
) -> Any:
    """The ranking step: action-cam candidates picked from the feature grid, then
    :func:`rank_parts`.

    Returns:
        :func:`rank_parts`'s events.json object.
    """
    steps.start("ranking")
    if cam is not None:
        from cutpoint import actioncam, activity  # noqa: PLC0415

        cam_ctx = cam.ctx
        if events_new:  # with this run's events.json and crashes.json
            cam_ctx = actioncam.prepare(sources, cam.durations, out_dir, quiet=True)
        # the beauty / people features on the captions of the project's activity.json
        # (written by the crash step above; action-cam without one)
        with activity.use_prompts(activity.project_prompts(out_dir)):
            for r in results:
                r.candidates = pick_candidates(r, style, opts.max_candidates, timings, cam_ctx)
                logger.info("%s: %d candidates", r.source.name, len(r.candidates))
        # a source's own set is known only once it is picked from
        hl["settings"]["features_by_source"] = _features_by_source(results)
    if opts.ids is not None:  # a re-rank's new spans get ids above the earlier analysis's
        hl["settings"]["last_candidate_id"] = opts.last_candidate_id
    return rank_parts(
        hl,
        results,
        style,
        out_dir,
        steps,
        t_start=t_start,
        crash_moments=opts.crash_moments,
        vision=opts.vision,
        vision_top=opts.vision_top,
        ollama_url=opts.ollama_url,
        ollama_model=opts.ollama_model,
        batch_size=opts.batch_size,
        vision_time_share=opts.vision_time_share,
        earlier_total_s=opts.earlier_total_s,
        ids=opts.ids,
    )


def _write_outputs(
    hl: dict[str, Any],
    ev: Any,
    results: list[SourceResult],
    transcript: str,
    timings: dict[str, float],
    t_start: float,
    earlier_total_s: float | None,
    prefetch: OcrPrefetch | None,
    held: Any,
    out_dir: Path,
) -> Path:
    """transcript.txt and highlights.json written and every :data:`HELD_OUTPUTS` file committed
    at once.

    Returns:
        Path of the written highlights.json.
    """
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
    out = out_dir / HIGHLIGHTS
    cache.save_json(out, backends.saved_settings(hl), durable=True)
    if ev:  # ranked with the events: a pre-events copy of an older analysis is stale
        cache.remove(out_dir / PRE_EVENTS)
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
    ids: Mapping[tuple[str, float, float], str] | None = None,
    vision_cache_only: bool = False,
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

    For a re-rank (:mod:`cutpoint.rerank`, :func:`~cutpoint.reanalyze.reanalyze`): ``ids`` maps the
    :func:`~cutpoint.reanalyze.span_key` of the earlier candidates to their ids; a candidate with
    the span of one keeps its id and the others get the next ids after
    ``settings.last_candidate_id`` (:func:`number_candidates`). None: every candidate is numbered
    from ``h001``. ``vision_cache_only``: the vision step takes the cached ratings only and asks no
    model (the moments without one stay unrated).

    Returns:
        The events.json object the candidates were ranked with, or a falsy value without one.
    """
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
    vision_step._vision_step(
        hl,
        ev,
        grids,
        style,
        project,
        steps,
        t_start=t_start,
        vision=vision,
        vision_top=vision_top,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        batch_size=batch_size,
        vision_time_share=vision_time_share,
        earlier_total_s=earlier_total_s,
        vision_cache_only=vision_cache_only,
    )

    cands = hl["candidates"]
    cands.sort(
        key=lambda c: (not c.get("must_include"), *scorer.rank_key(c), *scorer.tie_key(c))
    )  # must-include first
    suggested = len(default_fit.fit_target(cands, style.target_total))
    if ids is None:
        hl["candidates"] = [{"id": f"h{rank:03d}", **c} for rank, c in enumerate(cands, 1)]
        hl["settings"]["last_candidate_id"] = len(cands)
    else:  # unchanged spans keep their id, the others get the next ones
        given: set[str] = set()
        out = []
        for c in cands:
            was = ids.get(reanalyze.span_key(c))
            if was is not None and was not in given:
                given.add(was)
                c = {"id": was, **c}
            out.append(c)
        hl["candidates"] = out
        number_candidates(hl)
    hl["suggested_top"] = suggested
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
