"""``cutpoint analyze``: find and rank highlight candidates in one or more videos.

Signals per source (all on a 0.1 s grid, each normalised to 0-1):
    loudness  - robust z-score of ebur128 momentary loudness vs the clip's baseline
    speech    - style keywords/exclamations from the faster-whisper transcript
    laughter  - laughter tokens in the transcript
    scene     - density of PySceneDetect cuts (from a small, low-fps proxy)
    continuity- fraction of the surrounding ~10 s covered by speech (commentary)
The weighted sum is peak-picked into windows, merged, snapped to scene cuts and
word boundaries, optionally re-scored by a vision model, and written to
``highlights.json``.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cutpoint import cache, jsonout, progress
from cutpoint.config import remap_source, unc_path
from cutpoint.dsp import conv_same
from cutpoint.events import paths as event_paths
from cutpoint.events.menu_spans import drop_menu_time
from cutpoint.events.scoring import rescore
from cutpoint.events.setup import COMPONENTS, EVENTS_MODULES
from cutpoint.ffmpeg_utils import (
    READ_ERROR_RE,
    FFmpegError,
    HwDecoder,
    MediaInfo,
    ProgressResult,
    decoder_for,
    extract_audio,
    ffprobe_json,
    gpu_chain,
    momentary_loudness,
    output_dir,
    probe,
    read_window_ok,
    run_progress,
    single_video_stream,
    wav_duration,
)
from cutpoint.progress import ANALYZE_STEPS, LABELS, SKIPPABLE
from cutpoint.sources import READ_ERRORS_FILE, load_read_errors
from cutpoint.styles import LAUGHTER, PROFANITY, Style
from cutpoint.transcribe import (
    Segment,
    Transcriber,
    filter_hallucinations,
    flat_words,
    transcript_name,
)
from cutpoint.utils import clock, find_videos, matches_glob, slugify
from cutpoint.vision import VisionRater

if TYPE_CHECKING:
    from cutpoint.events.ocr_prefetch import OcrPrefetch

logger = logging.getLogger(__name__)

DT = 0.1  # analysis grid step (s), matches ebur128's 10 Hz momentary output
LONG_FILE_S = 1800  # above this, scene detection decodes keyframes only
PROFANE = {p.rstrip("*") for p in PROFANITY}  # names as keyword_hits reports them
PROFANITY_CUE_S = 3.0  # a profane word counts only if another cue is this close
PROFANITY_CUE_MIN = 0.8  # minimum weight of a keyword that counts as a cue
VISION_TOP_MIN, VISION_TOP_MAX = 10, 60  # auto --vision-top: 1 per 2 min, clamped
VISION_LABEL_MIN = 7.0  # --vision: from this rating, the model's label names the clip
SNIPPET_CHARS = 240  # a candidate's transcript snippet
UNSURE_P = 0.5  # transcript words Whisper heard with less confidence are listed as unsure
SHORTFALL_S, SHORTFALL_FRAC, SHORTFALL_FLOOR_S = 2.0, 0.01, 0.5  # truncation thresholds
RESUME_STEPS_S = (5, 10, 20, 40, 80, 160, 320)  # probe offsets past a read failure
EXTRACT_SHARE = 0.1  # the audio extract's share of a source in the transcribe step's pct
EXTRACT_POLL_S = 0.5  # how often the growing WAV is measured for that pct
WAV_BYTES_PER_S = 16000 * 2  # extract_audio's 16 kHz mono s16 output
WAV_HEADER = 44
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
    "telemetry.json",  # ski
    "ski_features",
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
    vision: dict[str, Any] | None = None
    score: float = 0.0
    label: str = ""
    kind: str = ""  # ski: dominant feature of the clip (airtime, crash, speed, ...)
    must_include: bool = False  # ski: confident crash (cutpoint clips always exports it)
    event: dict[str, Any] | None = None  # ski: the crash behind a must-include clip


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
    words: list[dict[str, Any]], patterns: dict[str, float]
) -> list[tuple[float, float, str]]:
    """Find (time, weight, phrase) matches; ``*`` suffix = prefix match."""
    compiled: dict[tuple[str, ...], tuple[float, str]] = {}
    for phrase, weight in patterns.items():
        parts = tuple(_norm_token(p) for p in phrase.split())
        compiled[parts] = (weight, phrase.rstrip("*"))
    toks = [_norm_token(w["word"]) for w in words]
    hits = []
    for i in range(len(toks)):
        for parts, (weight, phrase) in compiled.items():
            if i + len(parts) > len(toks):
                continue
            if all(
                toks[i + j].startswith(p[:-1]) if p.endswith("*") else toks[i + j] == p
                for j, p in enumerate(parts)
            ):
                hits.append((float(words[i]["start"]), weight, phrase))
    return hits


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
    wav: Path, cdir: Path, tr: Transcriber, on_progress: Callable[[float], None] | None = None
) -> list[Segment]:
    path = cdir / transcript_name(tr.model_name, tr.language)
    data = cache.load_json(path)
    if data is None:
        data = tr.transcribe(wav, on_progress) if on_progress else tr.transcribe(wav)
        cache.save_json(path, data)
    return data


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


def _proxy_args(
    duration: float, offset: float = 0.0, limit: float | None = None
) -> tuple[list[str], list[str], str, list[str]]:
    """The scene proxy's input options, output options, filter chain and encoder arguments."""
    long_file = duration > LONG_FILE_S
    pre = ["-skip_frame", "nokey"] if long_file else []
    if offset > 0:
        pre += ["-ss", f"{offset:.3f}"]
    post = ["-t", f"{limit:.3f}"] if limit else []
    vf = "fps=2,scale=256:-2" if long_file else "fps=10,scale=320:-2"
    return pre, post, vf, ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "32"]


def _detect(proxy: Path, path: Path) -> list[float]:
    """Scene cuts of ``proxy`` (deleted after), saved to ``path``."""
    from scenedetect import AdaptiveDetector, SceneManager, open_video  # noqa: PLC0415

    video = open_video(str(proxy))
    manager = SceneManager()
    manager.auto_downscale = False
    manager.add_detector(AdaptiveDetector(min_scene_len=5))
    manager.detect_scenes(video, show_progress=False)
    cuts = [round(s[0].seconds, 3) for s in manager.get_scene_list()[1:]]
    del video
    proxy.unlink(missing_ok=True)
    cache.save_json(path, cuts)
    return cuts


def scene_cuts(
    src: Path,
    cdir: Path,
    duration: float,
    offset: float = 0.0,
    limit: float | None = None,
    shared: Any = None,
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
    """
    path = cdir / "scenes.json"
    cached = cache.load_json(path)
    if cached is not None:
        return cached
    importlib.import_module("scenedetect")  # not installed: fails before any decode
    if shared is not None and (made := shared.result()) is not None:
        return _detect(made, path)
    proxy = cdir / "proxy.mp4"
    pre, post, vf, enc = _proxy_args(duration, offset, limit)
    label = f"scene proxy {src.name}"

    def command(dec: HwDecoder | None) -> list[str]:  # the decoder's input options go before -i
        chain = gpu_chain(dec, vf)
        args = [*pre, *(dec.input_args() if dec else []), "-i", str(src), *post]
        return ["ffmpeg", "-y", "-v", "error", *args, "-an", "-vf", chain, *enc, str(proxy)]

    hw = decoder_for(src)  # NVDEC or VAAPI when probed here; a failed GPU run decodes on the CPU
    try:
        run_progress(command(hw), duration, label)
    except FFmpegError:
        if hw is None:
            raise
        run_progress(command(None), duration, label)
    return _detect(proxy, path)


# --------------------------------------------------------------------------- windowing
def _pick_windows(
    src: Path,
    combined: np.ndarray,
    sig: dict[str, np.ndarray],
    style: Style,
    duration: float,
    max_candidates: int,
) -> list[Candidate]:
    score = combined.copy()
    gmax = float(score.max()) if score.size else 0.0
    out: list[Candidate] = []
    while len(out) < max_candidates:
        i = int(score.argmax())
        peak = float(score[i])
        if peak <= 0.05 or peak < 0.25 * gmax:
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


def _merge(cands: list[Candidate], style: Style) -> list[Candidate]:
    cands = sorted(cands, key=lambda c: c.start)
    merged: list[Candidate] = []
    for c in cands:
        prev = merged[-1] if merged else None
        if (
            prev
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
    return any(s["start"] - 0.3 <= t <= s["end"] + 0.3 for s in segments)


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
        near = [x for x in cuts if abs(x - t) <= 1.0]
        if near:
            setattr(c, attr, min(near, key=lambda x: abs(x - t)))
    if style.snap_sentences:
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
        elif max(c.end, prev.end) - prev.start <= style.max_len + min(15.0, 0.25 * style.max_len):
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


def vision_label(rating: float, label: str) -> str | None:
    """The model's ``label`` as a clip label, only for a rating of at least
    ``VISION_LABEL_MIN``; None keeps the clip's own label."""
    slug = slugify(label, 30) if label and rating >= VISION_LABEL_MIN else ""
    return slug or None


def _label(c: Candidate, style: Style) -> str:
    if c.vision and (slug := vision_label(c.vision.get("score", 0), c.vision.get("label", ""))):
        return slug
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
    cuts: list[float]


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
    """

    source: Path
    info: MediaInfo
    offset: float
    duration: float
    candidates: list[Candidate]
    segments: list[Segment]
    read_error: dict[str, Any] | None
    signals: PartSignals | None = None


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
        out.append({**seg, "start": seg["start"] + dt, "end": seg["end"] + dt, "words": words})
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
    ski_ctx: Any = None,
    pick: bool = True,
    report: Callable[[float], None] | None = None,
    read_checked: Callable[[MediaInfo, dict[str, Any] | None], Any] | None = None,
) -> SourceResult:
    """Compute signals for one source (or a recovered part starting at ``offset``).

    Internally everything runs in part-relative time; candidates and the
    transcript are returned in absolute source time. With ``ski_ctx`` (a
    :class:`cutpoint.ski.SkiContext`), candidates come from the ski
    feature grid instead of the generic signals. ``src`` is the original path
    (recorded everywhere); files are read through ``source_remap``. With ``pick``
    False no candidates are picked yet (:func:`pick_candidates` does it later).
    ``report`` gets the share of the source's transcription done (0-1): the audio
    extract is its first :data:`EXTRACT_SHARE`, Whisper the rest. ``read_checked`` gets
    the probe and the read-error record (None: a clean read) as soon as they are known,
    before loudness, Whisper and the scene cuts; what it returns goes to
    :func:`scene_cuts` as ``shared`` (a scene proxy made in the background, or None).
    """

    def tick(name: str, t0: float) -> None:
        timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - t0, 2)

    io = remap_source(src)  # read the copy, record the original
    info = probe(io)
    cdir = cache.source_dir(src, offset)
    span = info.duration - offset
    read_error: dict[str, Any] | None = None
    shared = None  # read_checked's background scene proxy
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

        if not wav.exists():
            with _watch_extract(wav, span, extracted if report else None):
                res = extract_audio(io, wav, info, start=offset)
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
        segments = _transcript(wav, cdir, tr, transcribed if report else None)
        tick("transcribe_s", t0)
        segments, dropped = filter_hallucinations(segments)
        if dropped:
            logger.info("%s: dropped %d likely-hallucinated transcript segments", src.name, dropped)
    else:
        logger.warning("%s has no audio; using scene/vision signals only", src.name)
        if read_checked is not None:
            shared = read_checked(info, None)

    n = max(1, int(span / DT) + 1)
    sig = {k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    if lufs_data is not None:
        times, lufs = lufs_data
        sig["loudness"] = np.interp(np.arange(n) * DT, times, loudness_signal(lufs))
    words = flat_words(segments)
    kw_hits = keyword_hits(words, style.keywords)
    laugh_hits = keyword_hits(words, LAUGHTER)
    if not style.profanity_alone:
        n_before = len(kw_hits)
        n_profane = sum(1 for *_, p in kw_hits if p in PROFANE)
        kw_hits = gate_profanity(kw_hits, laugh_hits, sig["loudness"])
        if n_profane:
            logger.info(
                "%s: ignored %d of %d profanity hits with no other cue",
                src.name,
                n_before - len(kw_hits),
                n_profane,
            )
    sig["speech"] = keyword_signal(n, kw_hits, style.keyword_cap)
    sig["laughter"] = _impulse_signal(n, [(t, w) for t, w, _ in laugh_hits])
    sig["continuity"] = continuity_signal(n, words)

    cuts: list[float] = []
    if use_scenes:
        t0 = time.perf_counter()
        cuts = scene_cuts(io, cdir, span, offset, limit=span if read_error else None, shared=shared)
        sig["scene"] = _impulse_signal(n, [(t, 0.7) for t in cuts], sigma_s=1.5)
        tick("scenes_s", t0)
        if shared is not None:  # for measuring the shared decode
            took = time.perf_counter() - t0
            logger.info(
                "%s: shared decode (%s): scenes %.1fs, %s",
                src.name,
                shared.mode,
                took,
                shared.summary(),
            )

    res = SourceResult(
        source=src,
        info=info,
        offset=offset,
        duration=span,
        candidates=[],
        segments=_shift_segments(segments, offset),
        read_error=read_error,
        signals=PartSignals(n, sig, kw_hits, laugh_hits, words, segments, cuts),
    )
    if pick:
        res.candidates = pick_candidates(res, style, max_candidates, timings, ski_ctx)
    where = f" from {clock(offset)}" if offset else ""
    logger.info(
        "%s%s: %.0fs, %d words, %d scene cuts%s",
        src.name,
        where,
        span,
        len(words),
        len(cuts),
        f", {len(res.candidates)} candidates" if pick else "",
    )
    return res


def pick_candidates(
    res: SourceResult,
    style: Style,
    max_candidates: int | None,
    timings: dict[str, float],
    ski_ctx: Any = None,
) -> list[Candidate]:
    """Candidates of one analysed part (:func:`analyze_source`), in absolute source time."""
    src, span, offset = res.source, res.duration, res.offset
    ps = res.signals
    if ps is None:
        raise ValueError(f"{src}: no signals to pick candidates from")
    n, sig, kw_hits, laugh_hits = ps.n, ps.sig, ps.kw_hits, ps.laugh_hits
    words, segments = ps.words, ps.segments
    if ski_ctx is not None:
        cands = _ski_candidates(
            src, n, span, sig, offset, style, ski_ctx, max_candidates, segments, timings
        )
    else:
        if max_candidates is None:  # enough to cover ~3x the target duration
            max_candidates = max(
                30, int(3 * style.target_total / ((style.min_len + style.max_len) / 2))
            )
        combined = sum(style.weights.get(k, 0.0) * v for k, v in sig.items())
        cands = _merge(_pick_windows(src, combined, sig, style, span, max_candidates), style)
        for c in cands:
            _snap(c, ps.cuts, segments, span, style)
        cands = _resolve_overlaps(cands, style)
    wsum = sum(style.weights.values()) or 1.0
    for c in cands:
        # A must-include crash scores its confidence (0-100), so it also ranks first.
        c.score = round(100 * (c.peak if c.must_include else c.peak / wsum), 1)
        inside = [(t, w, p) for t, w, p in kw_hits + laugh_hits if c.start <= t <= c.end]
        c.keywords = [p for _, _, p in sorted(inside, key=lambda h: -h[1])]
        c.keywords = list(dict.fromkeys(c.keywords))[:5]
        strong = [
            p for _, w, p in sorted(inside, key=lambda h: -h[1]) if w >= 0.5 and p not in PROFANE
        ]  # a bare swear is a useless label
        if not c.label_hint:  # ski picks bring their own ("jump 0.6s", "crash 7g")
            c.label_hint = strong[0] if strong else ""
        said = [w for w in words if c.start <= w["start"] <= c.end]
        c.transcript, c.transcript_unsure = _transcript_snippet(said)
        if offset:  # back to absolute source time: clips cut the original file
            c.start, c.end = round(c.start + offset, 3), round(c.end + offset, 3)
            c.peak_t += offset
    return cands


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


def _ski_candidates(
    src: Path,
    n: int,
    span: float,
    sig: dict[str, np.ndarray],
    offset: float,
    style: Style,
    ski_ctx: Any,
    max_candidates: int | None,
    segments: list[Segment],
    timings: dict[str, float],
) -> list[Candidate]:
    """Candidates from the ski feature grid (telemetry + audio + VLM tags)."""
    from cutpoint import ski  # noqa: PLC0415

    t0 = time.perf_counter()
    feats, mask, friends = ski.build_features(src, n, sig, ski_ctx, offset)
    tel = ski_ctx.telemetry.get(str(src))
    events = tel.events if tel is not None else []
    ski.save_features(ski_ctx.project, src, offset, span, feats, mask, friends, events)
    if max_candidates is None:  # short clips: a few; long ones: one per ~20 s
        max_candidates = max(4, int(span / 20))
    picks = ski.pick(
        feats,
        style.weights,
        mask,
        friends,
        events,
        span,
        max_candidates,
        offset,
        friends_strong=style.mask_friends,
    )
    forced = ski.must_include_picks(src, ski_ctx, style.must_include_crash, span, offset)
    if forced:
        logger.info(
            "%s: %d must-include crash clip(s) (confidence >= %.2f)",
            src.name,
            len(forced),
            style.must_include_crash,
        )
        picks = ski.merge_must_include(picks, forced)
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
    timings["ski_s"] = round(timings.get("ski_s", 0.0) + time.perf_counter() - t0, 2)
    return cands


def auto_vision_top(total_duration_s: float) -> int:
    """Default ``--vision-top``: one candidate per 2 minutes of footage, 10-60."""
    return max(VISION_TOP_MIN, min(VISION_TOP_MAX, math.ceil(total_duration_s / 120)))


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


# --------------------------------------------------------------------------- analysis steps
GAME_STYLES = ("gaming", "funny")  # keyframes: HUD OCR (with region discovery)
ACTION_STYLES = ("action", "sport", "ski")  # keyframes: CLIP; crash detection
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
LAYER_STEP = {
    "audio": "reactions",
    "voice": "reactions",
    "visual": "keyframes",
    "ocr": "keyframes",
    "crash": "crashes",
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


def has_telemetry(src: Path) -> bool:
    """Whether ``src`` carries camera telemetry (a GoPro GPMF track or an Insta360 X3 IMU
    trailer); only the container's stream list is read."""
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    try:
        if imu_source(src) is not None:
            return True
        info = ffprobe_json(remap_source(src), entries="stream=codec_tag_string", timeout=60)
    except (OSError, FFmpegError):
        return False
    return any(s.get("codec_tag_string") == "gpmd" for s in info.get("streams") or [])


def _cuda() -> bool:
    """Whether torch sees a CUDA GPU; False when torch itself fails to load (a broken
    install, a missing DLL), so the analysis goes on without the GPU-only steps."""
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        logger.debug("torch CUDA check failed", exc_info=True)
        return False


def plan_layers(style: Style, sources: Sequence[Path], skip: Collection[str]) -> LayerPlan:
    """Which event layers the reactions, keyframes and crashes steps run.

    Reactions: the PANNs audio events (plus the voice layer for action-cam footage, whose
    crash audio score reads it). Keyframes: HUD OCR for game styles, the CLIP visual layer
    for action footage (only with a CUDA GPU). Crashes: action-cam footage only (telemetry,
    or an action style). A step whose events extra or model files are missing is left out
    with one INFO line naming the fix; nothing is downloaded.
    """
    plan = LayerPlan(game=style.base in GAME_STYLES)
    action = style.kind == "ski" or style.base in ACTION_STYLES
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
            want["keyframes"] = ("ocr",)
        elif action:
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
    if "keyframes" in want and want["keyframes"] == ("visual",) and not _cuda():
        logger.info(
            "No CUDA GPU: keyframes skipped (CLIP on the CPU would take about an hour per "
            "hour of footage)"
        )
        plan.skipped["keyframes"] = SKIP_NO_CUDA
        del want["keyframes"]
    components: dict[str, None] = {}
    for step, layers in list(want.items()):
        files = event_paths.missing_files(layers)
        if not files:
            continue
        components.update(dict.fromkeys(comp for comp, _ in files))
        rest = tuple(x for x in layers if x != "voice")
        if "voice" in layers and not event_paths.missing_files(rest):
            want[step] = rest  # reactions without the voice layer
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


class _StepHook:
    """``run_events``'s ``stage`` for :func:`analyze`: each layer reports as its step; a
    layer that fails fails its step (an ERROR line and a ``step_failed`` warning), and the
    next steps still run."""

    def __init__(self, steps: progress.Run) -> None:
        self.steps = steps
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
        except Exception as err:
            message = f"{LABELS[step]} failed: {err}; continuing without it"
            logger.debug("%s layer failed", layer, exc_info=True)
            logger.error("%s", message)
            jsonout.warn("step_failed", message)
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
    style: Style, sources: Sequence[Path], skip: Collection[str], project: Path, layers: bool
) -> contextlib.AbstractContextManager[Any]:
    """The background OCR reader of :mod:`cutpoint.events.ocr_prefetch` when the keyframes
    step will run HUD OCR (the conditions of :func:`plan_layers`, checked without its log
    lines, which keep their place) and ``CUTPOINT_ANALYZE_OVERLAP`` is not ``0``; else a
    context giving None."""
    if not layers or style.base not in GAME_STYLES or "keyframes" in skip:
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
    return ocr_prefetch.OcrPrefetch(project, sources, OcrParams(game=game, player_tags=tags))


def _scene_plan(src: Path, duration: float) -> Any:
    """The :class:`~cutpoint.events.ocr_prefetch.ScenePlan` of a whole source whose scene
    proxy may be made in the background: up to :data:`LONG_FILE_S` (longer files decode
    keyframes only), its scene cuts not cached and one video stream (no cover art); else
    None (an INFO line names a failed stream gate; called in a shared-decode mode only)."""
    from cutpoint.events.ocr_prefetch import ScenePlan  # noqa: PLC0415

    cdir = cache.source_dir(src)
    if duration > LONG_FILE_S or cache.load_json(cdir / "scenes.json") is not None:
        return None
    io = remap_source(src)
    if not single_video_stream(io):
        logger.info(
            "%s: shared decode off: not exactly one video stream (cover art or several)",
            src.name,
        )
        return None
    _pre, _post, vf, enc = _proxy_args(duration)
    return ScenePlan(io, cdir, duration, vf, tuple(enc))


def _run_layers(
    project: Path,
    plan: LayerPlan,
    hl: dict[str, Any],
    steps: progress.Run,
    llm: bool,
    prefetch: OcrPrefetch | None = None,
    transcript: str | None = None,
) -> bool:
    """Run the planned event layers on the analysis ``hl`` (not written yet) and write
    events.json, as ``cutpoint events`` would (held back with the other
    :data:`HELD_OUTPUTS` until the analysis completes); False if they stopped before it.
    ``prefetch``: the background OCR reader (:func:`_ocr_prefetch`) or None; ``transcript``:
    the transcript.txt text, not written yet either (None: read from the project)."""
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
                ocr=OcrParams(game=game, player_tags=tags),
                vlm=VlmParams(max_windows=None),
                stage=_StepHook(steps),
                cached_ocr=plan.game,
                llm=llm,
                prefetch=prefetch,
            )
    except Exception as err:
        if steps.current == "ranking":
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
            "vision": c.vision["score"] if c.vision else None,
        },
        "label": _label(c, style),
        **({"kind": c.kind} if c.kind else {}),
        **({"must_include": True, "crash": c.event} if c.must_include else {}),
        "transcript": c.transcript,
        **({"transcript_unsure": c.transcript_unsure} if c.transcript_unsure else {}),
    }


def _rate_vision(
    cands: list[dict[str, Any]],
    rater: VisionRater,
    n_top: int,
    style: Style,
) -> tuple[bool, int]:
    """Rate the first ``n_top`` candidates with Ollama and apply the rating to their
    analysis score (``score_pre_events`` once events rescored them) with
    :func:`vision_blend` and the style's ``vision_weight``; the model's label replaces a candidate's from
    ``VISION_LABEL_MIN``. Returns ``(any rated, failures)``."""
    used, n_failed = False, 0
    top = cands[:n_top]
    try:
        for i, c in enumerate(top):
            progress.update(progress.fraction(i, len(top)))
            res = rater.rate(remap_source(Path(c["source"])), c["start"], c["end"], style.base)
            if res is None:
                n_failed += 1
                continue
            used = True
            c["signals"]["vision"] = res.score
            if res.label:
                c["vision_label"] = res.label
            if label := vision_label(res.score, res.label):
                c["label"] = label
            key = "score_pre_events" if "score_pre_events" in c else "score"
            c[key] = vision_blend(c[key], res.score, style.vision_weight)
            logger.info(
                "vision %.1f '%s' for %.1f-%.1f", res.score, res.label, c["start"], c["end"]
            )
    finally:  # free the VRAM now, not after Ollama's default 5 min keep_alive
        rater.unload()
    logger.info("Vision: %d ratings from the cache, %d asked the model", rater.hits, rater.asked)
    return used, n_failed


def analyze(
    target: Path | Sequence[Path],
    style: Style,
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
        style: Style preset.
        out_dir: Project directory.
        vision: True = rate the top candidates with Ollama (a warning if it is not
            available); False or None = no vision (opt-in, never automatic).
        model: faster-whisper model name (default: medium on CUDA, small on CPU).
        language: Force transcription language (e.g. "en"); None = autodetect.
        use_scenes: Run scene detection.
        max_candidates: Max candidates per source before merging (None = auto).
        vision_top: Number of top candidates to send to the vision model
            (None = ``auto_vision_top`` of the total source duration).
        ollama_url: Ollama base URL.
        ollama_model: Ollama vision model name.
        exclude: Glob patterns (file name or full path) to skip.
        recursive: Search sub-folders of folder inputs.
        recover: Resume after read errors by seeking past the bad region.
        layers: Run the reactions, keyframes and crashes steps and report the steps
            (``CUTPOINT_PROGRESS``); False (``reanalyze``, ``360 analyze``): reuse
            events.json as it is.
        skip: Steps of :data:`cutpoint.progress.SKIPPABLE` left out.

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
        )


def _analyze(
    target: Path | Sequence[Path],
    style: Style,
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
) -> Path:
    """:func:`analyze`, reporting to ``steps``."""
    t_start = time.perf_counter()
    steps.start("transcribe")
    found = _find_inputs(target, exclude, recursive)
    # Projects record the original paths; files are read through config source_remap.
    # De-duplicate by the file actually read; the first spelling given is recorded.
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
    with (
        output_dir(out_dir),  # a failed run leaves no empty project folder behind
        cache.hold_outputs(out_dir, HELD_OUTPUTS) as held,
        # HUD OCR in the background while the GPU steps run; stopped before the folder goes
        _ocr_prefetch(style, sources, skip, out_dir, layers) as prefetch,
    ):
        # ---- 1. transcribe: audio, loudness, Whisper and scene cuts per source
        use_scenes = use_scenes and style.use_scenes
        ski_ctx = None
        if style.kind == "ski":
            from cutpoint import ski  # noqa: PLC0415

            t0 = time.perf_counter()
            stale = out_dir / ski.FEATURES_DIR
            if stale.is_dir():
                for f in stale.glob("*.npz"):
                    cache.remove(f)  # held: removed only when the run completes
            taste = (style.learned or {}).get("taste")
            durations = {str(s): probe(remap_source(s)).duration for s in sources}
            ski_ctx = ski.prepare(sources, durations, out_dir, taste=Path(taste) if taste else None)
            telemetry_s = round(time.perf_counter() - t0, 2)
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
                if ok and use_scenes and prefetch.mode is not None:
                    plan = _scene_plan(src, info.duration)
                return prefetch.offer(src, round(info.duration, 3), ok, plan)

            return offer

        timings: dict[str, float] = {}
        if ski_ctx is not None:
            timings["telemetry_s"] = telemetry_s
        results: list[SourceResult] = []
        transcript_lines: list[str] = []
        for i, src in enumerate(sources):
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
                # Ski candidates come from the feature grid, after the event layers.
                res = analyze_source(
                    src,
                    style,
                    tr,
                    use_scenes,
                    max_candidates,
                    timings,
                    offset=offset,
                    recover=recover,
                    pick=ski_ctx is None,
                    report=report if steps.active else None,
                    read_checked=read_checked(src, offset),
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
        # Written with highlights.json at the end; the event layers read it from memory.
        transcript = "\n".join(transcript_lines)
        tr.close()  # Whisper out of memory before the next step loads a model
        free_models()
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
            "created": None,
            "settings": {
                **(
                    {
                        "mask_friends": style.mask_friends,
                        "must_include_crash": style.must_include_crash,
                    }
                    if style.kind == "ski"
                    else {}
                ),
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

        def records() -> list[dict[str, Any]]:
            """The candidates, best first; ``source_index`` is the index into ``results``."""
            found = [(c, i) for i, r in enumerate(results) for c in r.candidates]
            found.sort(key=lambda ci: -ci[0].score)
            return [_candidate_record(c, i, style) for c, i in found]

        # ---- 2-4. reactions, keyframes, crashes: the event layers (events.json)
        events_new = False
        if layers:
            plan = plan_layers(style, sources, skip)
            for step, reason in plan.skipped.items():
                steps.skip(step, reason)
            if plan.layers:
                hl["candidates"] = records()  # VLM windows cached by `events` follow them
                t0 = time.perf_counter()
                events_new = _run_layers(
                    out_dir,
                    plan,
                    hl,
                    steps,
                    llm=bool(vision),
                    prefetch=prefetch,
                    transcript=transcript,
                )
                timings["events_s"] = round(time.perf_counter() - t0, 2)
                free_models()

        # ---- 5. ranking: candidates rescored with the events
        steps.start("ranking")
        if ski_ctx is not None:
            if events_new:  # with this run's events.json and crashes.json
                ski_ctx = ski.prepare(
                    sources, durations, out_dir, taste=Path(taste) if taste else None
                )
            for r in results:
                r.candidates = pick_candidates(r, style, max_candidates, timings, ski_ctx)
                logger.info("%s: %d candidates", r.source.name, len(r.candidates))
        hl["candidates"] = records()
        ev = None
        if style.kind != "ski":  # ski: the events are features of the scorer itself
            ev = cache.load_json(out_dir / "events.json")
        if ev:
            drop_menu_time(hl, ev)
            changed = rescore(hl, ev, out_dir)
            logger.info(
                "Rescored %d of %d candidates from events.json", changed, len(hl["candidates"])
            )

        # ---- 6. vision: the Ollama rating, blended into the top candidates
        cands = hl["candidates"]
        if not vision:
            steps.skip("vision", "vision is off (Settings, config.json or --vision)")
        elif not cands:
            steps.skip("vision", "no candidates to rate")
        else:
            rater = VisionRater(
                **{k: v for k, v in {"url": ollama_url, "model": ollama_model}.items() if v}
            )
            # vision_top itself stays None (auto) in the saved settings, for reanalyze
            n_top = (
                auto_vision_top(sum(r.duration for r in results))
                if vision_top is None
                else vision_top
            )
            online, n_cached = rater.available(), 0
            if not online:  # Ollama down: the ratings already cached, no requests (#413)
                rater.cache_only = True
                n_cached = sum(
                    rater.cached(remap_source(Path(c["source"])), c["start"], c["end"], style.base)
                    is not None
                    for c in cands[:n_top]
                )
            if online or n_cached:
                if not online:
                    n_none = min(n_top, len(cands)) - n_cached
                    message = (
                        "vision rating requested but Ollama/model unavailable; using "
                        f"{n_cached} cached rating{'s' if n_cached != 1 else ''}, "
                        f"{n_none} candidate{'s' if n_none != 1 else ''} had no cached rating"
                    )
                    logger.warning("%s", message)
                    jsonout.warn("vision_unavailable", message)
                with steps.step("vision"):
                    t0 = time.perf_counter()
                    logger.info(
                        "Vision: rating top %d of %d candidates", min(n_top, len(cands)), len(cands)
                    )
                    used, n_failed = _rate_vision(cands, rater, n_top, style)
                    if ev:  # the blended analysis scores plus the event boosts again
                        rescore(hl, ev, out_dir)
                    timings["vision_s"] = round(time.perf_counter() - t0, 2)
                    hl["settings"]["vision_used"] = used
                    hl["settings"]["vision"] = {
                        "requested": min(n_top, len(cands)),
                        "failed": n_failed - rater.missed,
                        **({} if online else {"uncached": rater.missed}),
                    }
            else:
                message = (
                    "vision rating requested but Ollama/model unavailable; continuing without it"
                )
                logger.warning("%s", message)
                jsonout.warn("vision_unavailable", message)
                steps.skip("vision", "Ollama or its vision model is not reachable")

        cands = hl["candidates"]
        cands.sort(key=lambda c: (not c.get("must_include"), -c["score"]))  # must-include first
        total, suggested = 0.0, 0
        for c in cands:
            if total >= style.target_total and not c.get("must_include"):
                break
            total += c["end"] - c["start"]
            suggested += 1
        hl["candidates"] = [{"id": f"h{rank:03d}", **c} for rank, c in enumerate(cands, 1)]
        hl["suggested_top"] = suggested
        hl["created"] = datetime.now().isoformat(timespec="seconds")
        timings["total_s"] = round(time.perf_counter() - t_start, 2)
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
        logger.info("Wrote %d candidates to %s (%.1fs)", len(cands), out, timings["total_s"])
        report_highlights(out_dir, out)
        return out


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
    if "suggested_top" in saved:
        jsonout.output("suggested_top", saved["suggested_top"])
    jsonout.output("style", saved.get("style"))


def reanalyze(project: Path) -> Path:
    """Re-run ``analyze`` with the inputs and style recorded in highlights.json.

    Everything expensive is cached (audio, transcript, telemetry), so this is
    quick. ``cutpoint events`` calls it for ski projects, whose events feed the
    feature grid itself rather than a score boost.

    Raises:
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If highlights.json is unreadable, not valid JSON or records
            no analysis.
    """
    from cutpoint.styles import get_style  # noqa: PLC0415

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
    style = get_style(hl["style"])
    if st.get("must_include_crash") is not None:  # an `analyze --must-include-crash`
        style = replace(style, must_include_crash=float(st["must_include_crash"]))
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
