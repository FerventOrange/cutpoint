"""Development harness for #512: how often unreacted game audio (gunfire, explosions) takes
top-N slots in gaming footage, and whether a loudness variant cuts those without losing real
reactions. Not part of the package; default scoring is not changed. It reads Cutpoint's
per-source caches (WAV, loudness, transcript, scene cuts: :func:`cutpoint.analyze.analyze_source`
with ``pick=False``) and keeps its own data (per-track loudness, PANNs boom grids) under
``<cache>/loudness_eval/`` (or ``--cache-dir``), never under new keys in Cutpoint's caches.

Usage::

    uv run --offline python scripts/loudness_eval.py r6_446.mp4 warthunder_1.mp4 --out le.json
    uv run --offline python scripts/loudness_eval.py *.mp4 --variant boom:0.5 --variant boom:1
    uv run --offline python scripts/loudness_eval.py r6_1.mp4 --with-events r6-project
    uv run --offline python scripts/loudness_eval.py *.mp4 --draft-labels

Labels are ``<labels-dir>/<stem>.json`` (default ``labels/`` next to each source)::

    {"source": "r6_446.mp4", "labeller": "...",
     "moments": [{"t": 123.4, "kind": "reaction|game|both", "note": "..."}]}

``reaction``: a mic shout, laugh or callout; ``game``: an unreacted gunshot, explosion or kill
sound; ``both``: a game sound with a reaction within 3 s (counts as a reaction).

Variants (``--variant``, repeatable; ``baseline`` is always run; default: the plan's sweep
``voice boom:0.5 boom:0.75 boom:1 reweight:0.3 reweight:0.2``). Each changes only
``sig["loudness"]`` before :func:`cutpoint.analyze.pick_candidates`, or (``reweight``) only the
re-score after it:

- ``baseline``: today's scoring.
- ``voice``: loudness of the mic track alone (``voice-mic``, a1) and of
  ``max(mic, 0.5 x game)`` (``voice-max``, a2, in the 0-1 signal). The distinct tracks come from
  :func:`cutpoint.audio_tracks.kept_tracks` on :func:`cutpoint.ffmpeg_utils.decode_tracks`
  windows; the mic is the track that is loudest during transcript words relative to outside
  them. With one distinct track both are ``n/a`` and identical to the baseline.
- ``boom:K``: loudness times ``1 - K x boom``, where ``boom`` is the PANNs probability of the
  gunfire and explosion classes #580 removed, except within 1 s of speech (transcript words,
  and PANNs Laughter or Scream above their event thresholds). Skipped when PANNs or its model
  files are missing (lite); nothing is downloaded.
- ``reweight:W``: re-score only, after the events stage: candidates with OCR or VLM evidence in
  the events get gaming's loudness weight ``W`` (the score renormalised as a style with that
  weight would be), then their event bonus again. Needs ``--with-events``.

``--with-events PROJECT`` (repeatable) reads the project's ``events.json`` and adds a ``post``
stage (menu time dropped, events rescored, as ``analyze`` ranks) next to ``pre`` (the
analysis score).

Metrics per clip, variant and stage, at ``--top`` (default ``5 10 suggested``; ``suggested``
is the count of the default edit on that ranking, as ``suggested_top``):

- ``R@N``: reaction recall, each reaction label once, hit when its time is inside a top-N
  candidate;
- ``FP@N``: top-N candidates whose peak is within 1.5 s of a ``game`` label; ``mix``: those
  that also contain a ``reaction`` or ``both`` label (``fp_mixed``); ``G@N``: game labels
  inside a top-N candidate;
- ``+/-``: reactions gained and lost against the baseline of the same stage;
- ``loud%``: the loudness share of the top-``suggested`` peaks,
  ``w_loud x loudness[peak] / combined[peak]`` (mean and max).

Pooled totals follow, with the #512 decision criteria at ``suggested`` (FP down by at least
50%, no net loss, at most 1 lost per recording; the "3 recordings from 2 games" criterion is
for the reader). ``--out FILE`` writes everything as JSON. The JSON and tables name clips and
times: publish only the pooled numbers of local-only footage (r6, warthunder).

``--draft-labels`` writes a DRAFT ``<labels-dir>/<stem>.json`` per source for the owner to
correct (``"draft": true`` on the file and every moment, a ``note`` with the reason):
reactions from transcript laughter tokens, the style's exclamations and loud speech; game
moments from loudness peaks with no transcript word within 2 s, plus PANNs boom peaks when
PANNs is available (a reaction within 3 s of a boom peak becomes ``both``). Weaker moments are
added, strongest first, until each group has at least 30 pooled across the sources. An
existing label file is never overwritten without ``--force``.

``--model``, ``--language``, ``--no-scenes`` and ``--max-candidates`` are as on
``cutpoint analyze``: give the ones the analysis used, so its cached transcript and scene cuts
are found (a miss transcribes or detects again, as ``analyze`` would).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import json
import logging
import os
import sys
import wave
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import analyze, audio_tracks, cache, default_fit, features, scorer
from cutpoint.analyze import DT, PartSignals, SourceResult, loudness_signal
from cutpoint.config import remap_source, resolve_project_path
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import MediaInfo, decode_tracks, momentary_loudness
from cutpoint.styles import Style, get_style

logger = logging.getLogger(__name__)

KINDS = ("reaction", "game", "both")
REACTION_KINDS = ("reaction", "both")
GAME_PEAK_S = 1.5  # a candidate peaking this close to a game label is a game-only FP
SPEECH_GUARD_S = 1.0  # boom: no discount this close to speech
DRAFT_QUIET_S = 2.0  # draft game moments: no transcript word this close
DRAFT_BOTH_S = 3.0  # draft: a reaction this close to a boom peak is "both"
DRAFT_MERGE_S = 2.0  # draft reactions closer than this are one moment
DRAFT_PEAK_SEP_S = 3.0  # draft game peaks at least this far apart
MIN_DRAFTS = 30  # per group (reactions, game), pooled across the sources
LOUD_SPEECH, LOUD_SPEECH_FLOOR = 0.6, 0.4  # loudness signal of a "loud speech" word
GAME_PEAK, GAME_PEAK_FLOOR = 0.5, 0.3  # loudness signal of a draft game peak
BOOM_PEAK, BOOM_PEAK_FLOOR = 0.4, 0.25  # PANNs boom probability of a draft boom peak
KEYWORD_STRICT = 0.5  # keyword weight of a strict draft exclamation
VOICE_GAME_SHARE = 0.5  # voice-max: max(mic, 0.5 x game)
TRACK_CHUNK_S = 300.0  # per-track decode chunk
# The PANNs classes of the Gunfire and Explosion groups #580 removed (git show 67d153b).
BOOM_LABELS = (
    "Gunshot, gunfire",
    "Machine gun",
    "Fusillade",
    "Artillery fire",
    "Cap gun",
    "Explosion",
    "Boom",
)
VOICE_GROUP_NAMES = ("Laughter", "Scream")  # PANNs speech-like groups that block the discount
DEFAULT_VARIANTS = ("voice", "boom:0.5", "boom:0.75", "boom:1", "reweight:0.3", "reweight:0.2")
DEFAULT_TOP = ("5", "10", "suggested")
SUGGESTED = "suggested"
PANNS_MODULES = ("torch", "panns_inference", "torchlibrosa")
STAGES = ("pre", "post")


# --------------------------------------------------------------------------- variants
@dataclass(frozen=True)
class Variant:
    """One loudness variant: ``kind`` is baseline, voice-mic, voice-max, boom or reweight."""

    name: str
    kind: str
    param: float | None = None


BASELINE = Variant("baseline", "baseline")


def parse_variants(specs: list[str]) -> list[Variant]:
    """Variants of ``--variant`` specs, baseline first, without duplicates.

    Raises:
        ValueError: On an unknown spec or a bad ``K``/``W``.
    """
    out = [BASELINE]
    for spec in specs:
        name, _, arg = spec.partition(":")
        if name == "baseline" and not arg:
            continue
        if name == "voice" and not arg:
            found = [Variant("voice-mic", "voice-mic"), Variant("voice-max", "voice-max")]
        elif name in ("boom", "reweight") and arg:
            try:
                value = float(arg)
            except ValueError:
                raise ValueError(f"{spec!r}: {name}:<number> expected") from None
            if value < 0 or (name == "reweight" and value > 1):
                raise ValueError(f"{spec!r}: out of range")
            found = [Variant(f"{name}:{value:g}", name, value)]
        else:
            raise ValueError(f"{spec!r}: expected baseline, voice, boom:K or reweight:W")
        out += [v for v in found if v not in out]
    return out


def parse_tops(specs: list[str]) -> list[str]:
    """``--top`` values as keys (``"5"``, ``"suggested"``), without duplicates."""
    out: list[str] = []
    for s in specs:
        key = SUGGESTED if s == SUGGESTED else str(int(s))
        if key != SUGGESTED and int(key) < 1:
            raise ValueError(f"--top {s}: must be at least 1")
        if key not in out:
            out.append(key)
    return out


def words_mask(n: int, words: list[dict[str, Any]], guard: float) -> np.ndarray:
    """True on the analysis grid within ``guard`` s of a spoken word."""
    mask = np.zeros(n, dtype=bool)
    for w in words:
        a = max(0, int((w["start"] - guard) / DT))
        b = min(n, int(np.ceil((w["end"] + guard) / DT)) + 1)
        mask[a:b] = True
    return mask


def dilate(mask: np.ndarray, seconds: float) -> np.ndarray:
    """``mask`` widened by ``seconds`` on each side."""
    r = round(seconds / DT)
    if r <= 0 or not mask.any():
        return mask.copy()
    k = np.ones(2 * r + 1)
    return np.convolve(mask.astype(float), k, mode="same") > 0


def boom_discount(loud: np.ndarray, boom: np.ndarray, speech: np.ndarray, k: float) -> np.ndarray:
    """``loud`` times ``1 - k x boom`` (at least 0) where ``speech`` is False."""
    factor = np.clip(1.0 - k * boom, 0.0, 1.0)
    return loud * np.where(speech, 1.0, factor)


def pick_mic(track_lufs: list[np.ndarray], talk: np.ndarray) -> int | None:
    """Index of the track with the most speech: the largest median loudness during ``talk``
    over the median outside it; None without both talk and silence to compare."""
    if not talk.any() or talk.all():
        return None
    gains = [
        float(np.median(np.maximum(x[talk], -70)) - np.median(np.maximum(x[~talk], -70)))
        for x in track_lufs
    ]
    return int(np.argmax(gains))


def voice_signal(mic: np.ndarray, game: list[np.ndarray], mode: str) -> np.ndarray:
    """The voice variants' loudness signal: ``mic`` (voice-mic), or
    ``max(mic, 0.5 x the loudest game track)`` (voice-max)."""
    if mode == "voice-mic" or not game:
        return mic.copy()
    return np.maximum(mic, VOICE_GAME_SHARE * np.max(np.stack(game), axis=0))


def peak_index(t: float, n: int) -> int:
    """Grid index of a candidate peak at ``t`` (part time)."""
    return min(max(round(t / DT), 0), n - 1)


def loud_share(w_loud: float, loud: float, combined: float) -> float:
    """``w_loud x loud / combined`` (0 for a zero peak)."""
    return round(w_loud * loud / combined, 3) if combined > 0 else 0.0


def to_records(
    cands: list[analyze.Candidate], sig: dict[str, np.ndarray], style: Style, source: str
) -> list[dict[str, Any]]:
    """highlights.json-like records of ``cands`` (as analyze writes them) plus ``peak_t``,
    ``peak`` (the combined grid there), ``loud_peak`` and ``loud_share``."""
    comb = scorer.combined(sig, style)
    w = style.weights.get("loudness", 0.0)
    out = []
    for c in cands:
        i = peak_index(c.peak_t, len(comb))
        loud = float(sig["loudness"][i])
        rec = analyze._candidate_record(c, 0, style)
        rec.update(
            source=source,
            peak_t=round(c.peak_t, 2),
            peak=float(c.peak),
            loud_peak=loud,
            loud_share=loud_share(w, loud, float(comb[i])),
        )
        out.append(rec)
    return out


def suggested_top(ranked: list[dict[str, Any]], target_total: float) -> int:
    """How many of ``ranked`` the default edit picks in ``target_total`` seconds, as analyze counts them."""
    return len(default_fit.fit_target(ranked, target_total))


def rank_pre(records: list[dict[str, Any]], target_total: float) -> tuple[list[dict], int]:
    """The analysis ranking: best score first, and its ``suggested_top``."""
    ranked = sorted(records, key=lambda c: -c["score"])
    return ranked, suggested_top(ranked, target_total)


def has_evidence(c: dict[str, Any]) -> bool:
    """Whether a rescored candidate has OCR (HUD event counts) or VLM evidence."""
    ev = (c.get("signals") or {}).get("events") or {}
    if ev.get("vlm"):
        return True
    return any(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
        for k, v in ev.items()
        if k not in ("boost", "laughter", "scream", "vlm")
    )


def reweight(
    ranked: list[dict[str, Any]],
    style: Style,
    w: float,
    target_total: float,
    avail: frozenset[str] | None,
) -> tuple[list[dict[str, Any]], int, int]:
    """(c1) re-score only: candidates with OCR or VLM evidence get the loudness weight ``w``
    (their peak and the score's divisor as a style with that weight), then their event bonus
    again. Returns the re-ranked copies, their suggested_top and how many changed weight."""
    weights = {**style.weights, "loudness": w}
    w0 = style.weights.get("loudness", 0.0)
    out, n = [], 0
    for c in copy.deepcopy(ranked):
        if has_evidence(c):
            n += 1
            peak = c["peak"] - (w0 - w) * c["loud_peak"]
            base = features.to_score(peak, weights, avail)
            ev = c["signals"]["events"]
            vlm = ev.get("vlm")
            vscore = None
            if vlm:
                vscore = max(vlm.get("highlight") or 0, vlm.get("funny") or 0) * 10
            c["score_pre_events"] = base
            c["score"] = scorer.with_bonus(base, ev.get("boost", 0.0), vscore)
            c["loud_share"] = loud_share(w, c["loud_peak"], peak)
        out.append(c)
    out.sort(key=lambda c: -c["score"])
    return out, suggested_top(out, target_total), n


# --------------------------------------------------------------------------- metrics
def load_labels(path: Path, source: Path) -> list[dict[str, Any]] | None:
    """The moments of a label file (None if there is none).

    Raises:
        ValueError: On a moment without a number ``t`` or with an unknown ``kind``.
    """
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source") and Path(str(data["source"])).name != source.name:
        logger.warning("%s names %s, not %s", path, data["source"], source.name)
    moments = []
    for m in data.get("moments", []):
        if not isinstance(m.get("t"), (int, float)) or m.get("kind") not in KINDS:
            raise ValueError(f"{path}: bad moment {m!r} (t: seconds, kind: {'|'.join(KINDS)})")
        moments.append({**m, "t": float(m["t"])})
    drafts = sum(1 for m in moments if m.get("draft")) or (len(moments) if data.get("draft") else 0)
    if drafts:
        logger.warning("%s: %d moment(s) still marked draft (not corrected yet)", path, drafts)
    return moments


def top_metrics(ranked: list[dict[str, Any]], n: int, labels: list[dict[str, Any]]) -> dict:
    """Recall, game-only FP and game reach of the top ``n`` of ``ranked``; ``fp_mixed`` counts
    the FP candidates that also contain a reaction (or ``both``) label."""
    top = ranked[:n]
    reactions = [i for i, m in enumerate(labels) if m["kind"] in REACTION_KINDS]
    games = [i for i, m in enumerate(labels) if m["kind"] == "game"]

    def inside(t: float) -> bool:
        return any(c["start"] <= t <= c["end"] for c in top)

    hit = [i for i in reactions if inside(labels[i]["t"])]
    fps = [c for c in top if any(abs(c["peak_t"] - labels[i]["t"]) <= GAME_PEAK_S for i in games)]
    fp = len(fps)
    fp_mixed = sum(
        1 for c in fps if any(c["start"] <= labels[i]["t"] <= c["end"] for i in reactions)
    )
    reached = sum(1 for i in games if inside(labels[i]["t"]))
    return {
        "n": len(top),
        "reactions": len(reactions),
        "hit": len(hit),
        "hit_ids": hit,
        "recall": round(len(hit) / len(reactions), 3) if reactions else None,
        "fp": fp,
        "fp_rate": round(fp / len(top), 3) if top else None,
        "fp_mixed": fp_mixed,
        "games": len(games),
        "game_reached": reached,
        "game_share": round(reached / len(games), 3) if games else None,
    }


def evaluate(
    ranked: list[dict[str, Any]], suggested: int, labels: list[dict[str, Any]], tops: list[str]
) -> dict[str, Any]:
    """Metrics of one ranking at every ``tops`` key, plus the loudness share of the top
    ``suggested`` candidates."""
    at = {k: top_metrics(ranked, suggested if k == SUGGESTED else int(k), labels) for k in tops}
    shares = [c.get("loud_share", 0.0) for c in ranked[:suggested]]
    top = [
        {k: c[k] for k in ("start", "end", "peak_t", "score", "loud_share") if k in c}
        for c in ranked[: max(suggested, *(int(k) for k in tops if k != SUGGESTED), 0)]
    ]
    return {
        "candidates": len(ranked),
        "suggested": suggested,
        "at": at,
        "loud_share": {
            "mean": round(float(np.mean(shares)), 3) if shares else None,
            "max": round(float(np.max(shares)), 3) if shares else None,
        },
        "top": top,
    }


def compare(result: dict[str, Any], base: dict[str, Any]) -> None:
    """Add reactions ``gained`` and ``lost`` against ``base`` to each top key of ``result``."""
    for k, m in result["at"].items():
        b = set(base["at"][k]["hit_ids"])
        v = set(m["hit_ids"])
        m["gained"], m["lost"] = len(v - b), len(b - v)


def pool(clips: list[dict[str, Any]], variants: list[str], tops: list[str]) -> dict[str, Any]:
    """Pooled totals per variant and stage, with the decision criteria against baseline."""
    out: dict[str, Any] = {}
    for name in variants:
        for stage in STAGES:
            rows = [
                c["variants"][name][stage] for c in clips if stage in c["variants"].get(name, {})
            ]
            if not rows:
                continue
            at: dict[str, Any] = {}
            for k in tops:
                ms = [r["at"][k] for r in rows]
                hit, reactions = sum(m["hit"] for m in ms), sum(m["reactions"] for m in ms)
                fp, slots = sum(m["fp"] for m in ms), sum(m["n"] for m in ms)
                reached, games = sum(m["game_reached"] for m in ms), sum(m["games"] for m in ms)
                at[k] = {
                    "hit": hit,
                    "reactions": reactions,
                    "recall": round(hit / reactions, 3) if reactions else None,
                    "fp": fp,
                    "slots": slots,
                    "fp_rate": round(fp / slots, 3) if slots else None,
                    "fp_mixed": sum(m["fp_mixed"] for m in ms),
                    "game_reached": reached,
                    "games": games,
                    "gained": sum(m.get("gained", 0) for m in ms),
                    "lost": sum(m.get("lost", 0) for m in ms),
                    "max_lost": max((m.get("lost", 0) for m in ms), default=0),
                }
            out.setdefault(name, {})[stage] = {
                "recordings": len(rows),
                "n_a": sum(1 for r in rows if r.get("na")),
                "at": at,
            }
    key = SUGGESTED if SUGGESTED in tops else tops[-1]
    for name, stages in out.items():
        if name == BASELINE.name:
            continue
        for stage, p in stages.items():
            base = out.get(BASELINE.name, {}).get(stage)
            if base is None:
                continue
            p["criteria"] = criteria(p["at"][key], base["at"][key], key)
    return out


def criteria(var: dict[str, Any], base: dict[str, Any], key: str) -> dict[str, Any]:
    """#512's adoption criteria at top ``key``: game-only FP down by at least 50%, no net
    pooled loss of reactions, at most one lost per recording."""
    drop = None if not base["fp"] else round(1 - var["fp"] / base["fp"], 3)
    net = var["gained"] - var["lost"]
    ok = {
        "fp_drop_50": drop is not None and drop >= 0.5,
        "no_net_loss": net >= 0,
        "max_1_lost_per_recording": var["max_lost"] <= 1,
    }
    return {"top": key, "fp_drop": drop, "net": net, **ok, "pass": all(ok.values())}


# --------------------------------------------------------------------------- signals (I/O)
def eval_cache(args_dir: Path | None) -> Path:
    """The script's own cache folder: ``--cache-dir`` or ``<cache>/loudness_eval``."""
    folder = args_dir if args_dir is not None else cache.cache_root() / "loudness_eval"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def source_cache(root: Path, src: Path) -> Path:
    """This source's folder in the script cache (named like its Cutpoint cache folder, which
    is keyed by path, size and modification time)."""
    folder = root / cache.source_dir(src).name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def distinct_tracks(io: Path, streams: int, duration: float, folder: Path) -> tuple[int, ...]:
    """:func:`cutpoint.audio_tracks.kept_tracks` of the source's compare windows (cached in the
    script's ``folder``, not in Cutpoint's ``audio-tracks.json``); every track when no window
    decodes (not cached)."""
    cfile = folder / "tracks.json"
    hit = cache.load_json(cfile)
    if hit and hit.get("streams") == streams:
        return tuple(hit["tracks"])
    length = min(audio_tracks.WINDOW_S, duration)
    windows = []
    for t in audio_tracks.window_starts(duration):
        try:
            pcm = decode_tracks(
                io, streams, sr=audio_tracks.SR, start=t, duration=length, timeout=60
            )
        except FFmpegError as err:
            logger.debug("track window %.1fs of %s failed: %s", t, io.name, err)
            continue
        if pcm.shape[1] >= audio_tracks.MIN_WINDOW_S * audio_tracks.SR:
            windows.append(pcm.astype(np.float64) / 32768.0)
    if not windows:
        logger.debug("track compare of %s: no window decoded; every track kept", io.name)
        return tuple(range(streams))
    kept = audio_tracks.kept_tracks(windows)
    cache.save_json(cfile, {"streams": streams, "tracks": list(kept)})
    return kept


def track_lufs(
    io: Path, streams: int, kept: tuple[int, ...], duration: float, folder: Path
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Momentary loudness ``(t, LUFS)`` of each ``kept`` track at 16 kHz, cached in ``folder``
    (the decoded WAVs are removed once measured)."""
    out = []
    todo = [i for i in kept if cache.load_json(folder / f"track{i}-loudness.json") is None]
    if todo:
        wavs = {i: folder / f"track{i}.wav" for i in todo}
        try:  # the decoded WAVs never stay, also when a decode fails
            with contextlib.ExitStack() as stack:
                handles = {i: stack.enter_context(wave.open(str(p), "wb")) for i, p in wavs.items()}
                for h in handles.values():
                    h.setnchannels(1)
                    h.setsampwidth(2)
                    h.setframerate(16000)
                t = 0.0
                while t < duration:
                    span = min(TRACK_CHUNK_S, duration - t)
                    pcm = decode_tracks(
                        io, streams, sr=16000, start=t, duration=span, timeout=max(60.0, span)
                    )
                    for i, h in handles.items():
                        h.writeframes(np.ascontiguousarray(pcm[i]).astype("<i2").tobytes())
                    t += TRACK_CHUNK_S
            for i, p in wavs.items():
                times, values = momentary_loudness(p)
                cache.save_json(folder / f"track{i}-loudness.json", {"t": times, "m": values})
        finally:
            for p in wavs.values():
                p.unlink(missing_ok=True)
    for i in kept:
        data = cache.load_json(folder / f"track{i}-loudness.json")
        out.append((np.asarray(data["t"]) - 0.2, np.asarray(data["m"])))  # as analyze shifts
    return out


def voice_inputs(
    res: SourceResult, folder: Path
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]] | str:
    """The mic and game loudness signals of a source on its analysis grid, and track info; or
    why the voice variants do not apply (``n/a``)."""
    info: MediaInfo = res.info
    if info.audio_streams < 2:
        return "single track"
    io = remap_source(res.source)
    try:
        kept = distinct_tracks(io, info.audio_streams, info.duration, folder)
    except FFmpegError as err:
        return f"track decode failed: {err}"
    if len(kept) < 2:
        return f"single distinct track ({info.audio_streams} tracks are copies)"
    ps = res.signals
    assert ps is not None
    grid = np.arange(ps.n) * DT
    try:
        measured = track_lufs(io, info.audio_streams, kept, res.duration, folder)
    except FFmpegError as err:
        return f"track decode failed: {err}"
    mic = pick_mic([np.interp(grid, t, m) for t, m in measured], words_mask(ps.n, ps.words, 0.0))
    if mic is None:
        return "no transcript words to tell the mic track by"
    sigs = [np.interp(grid, t, loudness_signal(m)) for t, m in measured]
    game = [s for j, s in enumerate(sigs) if j != mic]
    return sigs[mic], game, {"tracks": info.audio_streams, "kept": list(kept), "mic": kept[mic]}


def panns_missing() -> str | None:
    """Why PANNs cannot run here (None when it can)."""
    gone = [m for m in PANNS_MODULES if importlib.util.find_spec(m) is None]
    if gone:
        return f"PANNs is not installed (Cutpoint lite: no {', '.join(gone)})"
    from cutpoint.events import paths as event_paths  # noqa: PLC0415

    files = event_paths.missing_files(("audio",))
    if files:
        names = ", ".join(str(p) for _, p in files)
        return f"PANNs model files missing ({names}; cutpoint setup --only panns)"
    return None


class BoomGrids:
    """PANNs pooled probabilities of the boom and speech-like classes per source, cached in
    the script cache; the detector loads on the first miss."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.det: Any = None

    def classes(self) -> tuple[list[str], dict[str, float]]:
        """The class labels to pool and the abs threshold of each speech-like label."""
        from cutpoint.events import audio_events  # noqa: PLC0415

        thr: dict[str, float] = {}
        for g in audio_events.DEFAULT_GROUPS:
            if g.name in VOICE_GROUP_NAMES:
                thr.update(dict.fromkeys(g.labels, g.abs_thr))
        return [*BOOM_LABELS, *thr], thr

    def get(self, src: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``(boom, speech)`` of ``src`` on an ``n``-step analysis grid: the max boom class
        probability, and where a Laughter or Scream class reaches its event threshold."""
        from cutpoint.events import audio_events, common  # noqa: PLC0415

        labels, thr = self.classes()
        params = audio_events.AudioParams()
        source = common.load_source(src)
        wavs = [
            [str(p.wav), p.wav.stat().st_size, p.wav.stat().st_mtime_ns] for p in source.audio_parts
        ]
        key = cache.params_hash(
            {
                "labels": labels,
                "pool_s": params.pool_s,
                "block_s": params.block_s,
                "overlap_s": params.overlap_s,
                "wavs": wavs,
            }
        )
        path = source_cache(self.root, src) / f"boom-{key}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                probs, dt = np.asarray(data["probs"]), float(data["dt"])
        else:
            if self.det is None:
                self.det = audio_events.PannsDetector()
            names = audio_events.load_labels()
            idx = [names.index(lab) for lab in labels]
            probs, _ = audio_events.pooled_probabilities(source, self.det, params, idx)
            dt = max(1, round(params.pool_s / audio_events.FRAME_S)) * audio_events.FRAME_S
            np.savez_compressed(path, probs=probs, dt=np.float64(dt))
        return boom_speech(probs, dt, labels, thr, n)

    def close(self) -> None:
        if self.det is not None:
            self.det.close()
            self.det = None


def boom_speech(
    probs: np.ndarray, dt: float, labels: list[str], thr: dict[str, float], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled PANNs ``probs`` (row ``r`` covers ``[r dt, (r + 1) dt)``, a column per ``labels``)
    as the boom probability and the speech-like mask on an ``n``-step analysis grid (each step
    takes its row's value, so pooled peaks keep their height)."""
    if not len(probs):
        return np.zeros(n), np.zeros(n, dtype=bool)
    rows = np.minimum((np.arange(n) * DT / dt).astype(np.int64), len(probs) - 1)
    boom_cols = [i for i, lab in enumerate(labels) if lab in BOOM_LABELS]
    boom = probs[:, boom_cols].max(axis=1) if boom_cols else np.zeros(len(probs))
    voiced = np.zeros(len(probs), dtype=bool)
    for i, lab in enumerate(labels):
        if lab in thr:
            voiced |= probs[:, i] >= thr[lab]
    return boom[rows].astype(np.float64), voiced[rows]


# --------------------------------------------------------------------------- per source
def variant_signals(
    v: Variant,
    ps: PartSignals,
    voice: tuple[np.ndarray, list[np.ndarray], dict[str, Any]] | str | None,
    boom: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], str | None]:
    """The signals of variant ``v`` (only ``loudness`` differs) and its n/a reason (then the
    signals are the baseline's)."""
    sig = dict(ps.sig)
    if v.kind in ("voice-mic", "voice-max"):
        if isinstance(voice, str) or voice is None:
            return sig, voice or "no track data"
        mic, game, _ = voice
        sig["loudness"] = voice_signal(mic, game, v.kind)
    elif v.kind == "boom":
        assert boom is not None and v.param is not None
        b, voiced = boom
        speech = dilate(words_mask(ps.n, ps.words, 0.0) | voiced, SPEECH_GUARD_S)
        sig["loudness"] = boom_discount(ps.sig["loudness"], b, speech, v.param)
    return sig, None


def rank_post(
    records: list[dict[str, Any]],
    ev: dict[str, Any],
    project: Path,
    style: Style,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """The events ranking analyze makes: menu time dropped, then the events' bonus stage."""
    hl = {
        "style": style.name,
        "settings": {**settings, "target_total": style.target_total},
        "candidates": copy.deepcopy(records),
        "suggested_top": 0,
    }
    analyze.drop_menu_time(hl, ev)
    analyze.rescore(hl, ev, project)
    return hl["candidates"], hl["suggested_top"]


def events_for(
    src: Path, loaded: list[tuple[Path, dict[str, Any]]]
) -> tuple[Path, dict, str] | None:
    """The first ``(project, events.json, source spelling)`` that lists ``src``."""
    want = src.resolve()
    for project, ev in loaded:
        for s in ev.get("sources") or []:
            try:
                if Path(s).resolve() == want:
                    return project, ev, s
            except OSError:
                continue
    return None


def run_source(
    src: Path,
    labels: list[dict[str, Any]],
    res: SourceResult,
    style: Style,
    variants: list[Variant],
    tops: list[str],
    events: tuple[Path, dict, str] | None,
    settings: dict[str, Any],
    voice: Any,
    boom: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    """Every variant and stage of one analysed source."""
    ps = res.signals
    assert ps is not None
    name = events[2] if events else str(src)
    avail = scorer.available_for(style)
    out: dict[str, dict[str, Any]] = {}
    for v in variants:
        if v.kind == "reweight":
            continue
        sig, na = variant_signals(v, ps, voice, boom)
        part = replace(res, signals=replace(ps, sig=sig))
        cands = analyze.pick_candidates(part, style, settings["max_candidates"], {})
        records = to_records(cands, sig, style, name)
        stages = {}
        ranked, n = rank_pre(records, style.target_total)
        stages["pre"] = {**evaluate(ranked, n, labels, tops), "na": na}
        if events:
            ranked_post, n_post = rank_post(records, events[1], events[0], style, settings)
            stages["post"] = {**evaluate(ranked_post, n_post, labels, tops), "na": na}
            stages["post"]["_ranked"] = ranked_post
        out[v.name] = stages
    base_post = out[BASELINE.name].get("post")
    for v in variants:
        if v.kind != "reweight":
            continue
        if base_post is None:
            continue
        assert v.param is not None
        ranked, n, changed = reweight(
            base_post["_ranked"], style, v.param, style.target_total, avail
        )
        out[v.name] = {
            "post": {**evaluate(ranked, n, labels, tops), "na": None, "reweighted": changed}
        }
    for stages in out.values():
        for stage, r in stages.items():
            r.pop("_ranked", None)
            if stage in out[BASELINE.name]:
                compare(r, out[BASELINE.name][stage])
    return out


# --------------------------------------------------------------------------- draft labels
def local_peaks(x: np.ndarray, floor: float, allowed: np.ndarray, sep_s: float) -> list[int]:
    """Indices of local maxima of ``x`` at least ``floor``, where ``allowed``, strongest first
    and at least ``sep_s`` apart."""
    if len(x) < 3:
        return []
    i = np.flatnonzero((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= floor)) + 1
    i = i[allowed[i]]
    sep = round(sep_s / DT)
    out: list[int] = []
    for j in i[np.argsort(-x[i], kind="stable")]:
        if all(abs(int(j) - k) >= sep for k in out):
            out.append(int(j))
    return out


def suppress(items: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    """The strongest of ``items`` closer than ``gap`` s, in time order."""
    kept: list[dict[str, Any]] = []
    for it in sorted(items, key=lambda d: -d["strength"]):
        if all(abs(it["t"] - k["t"]) >= gap for k in kept):
            kept.append(it)
    return sorted(kept, key=lambda d: d["t"])


def draft_moments(
    ps: PartSignals, boom: tuple[np.ndarray, np.ndarray] | None = None
) -> list[dict[str, Any]]:
    """Draft moments of one source: ``t``, ``kind``, ``note``, ``strength`` (for ordering) and
    ``strict`` (above the first-pass threshold; the weaker ones fill up to MIN_DRAFTS)."""
    loud = ps.sig["loudness"]
    n = ps.n
    reactions: list[dict[str, Any]] = []
    for t, w, p in ps.laugh_hits:
        reactions.append(
            {"t": t, "note": f"laughter token {p!r}", "strength": 1.0 + w, "strict": True}
        )
    for t, w, p in ps.kw_hits:
        reactions.append(
            {"t": t, "note": f"exclamation {p!r}", "strength": w, "strict": w >= KEYWORD_STRICT}
        )
    for wd in ps.words:
        t = (wd["start"] + wd["end"]) / 2
        level = float(loud[peak_index(t, n)])
        if level >= LOUD_SPEECH_FLOOR:
            text = wd["word"].strip()
            reactions.append(
                {
                    "t": t,
                    "note": f"loud speech {text!r} (loudness {level:.2f})",
                    "strength": level,
                    "strict": level >= LOUD_SPEECH,
                }
            )
    reactions = [{**r, "kind": "reaction"} for r in suppress(reactions, DRAFT_MERGE_S)]
    quiet = ~words_mask(n, ps.words, DRAFT_QUIET_S)
    game: list[dict[str, Any]] = []
    for i in local_peaks(loud, GAME_PEAK_FLOOR, quiet, DRAFT_PEAK_SEP_S):
        game.append(
            {
                "t": i * DT,
                "note": f"loudness peak {loud[i]:.2f}, no words within {DRAFT_QUIET_S:g} s",
                "strength": float(loud[i]),
                "strict": loud[i] >= GAME_PEAK,
            }
        )
    if boom is not None:
        b = boom[0]
        everywhere = np.ones(n, dtype=bool)
        for i in local_peaks(b, BOOM_PEAK_FLOOR, everywhere, DRAFT_PEAK_SEP_S):
            t = i * DT
            near = [r for r in reactions if abs(r["t"] - t) <= DRAFT_BOTH_S]
            if near:
                for r in near:
                    r["kind"] = "both"
                    r["note"] += f"; boom peak {b[i]:.2f} at {t:.1f}s"
            elif quiet[i]:
                game.append(
                    {
                        "t": t,
                        "note": f"PANNs boom peak {b[i]:.2f}, no words within {DRAFT_QUIET_S:g} s",
                        "strength": float(b[i]),
                        "strict": b[i] >= BOOM_PEAK,
                    }
                )
    game = [{**g, "kind": "game"} for g in suppress(game, GAME_PEAK_S)]
    return sorted(reactions + game, key=lambda d: d["t"])


def select_drafts(
    per_source: dict[str, list[dict[str, Any]]], minimum: int = MIN_DRAFTS
) -> dict[str, list[dict[str, Any]]]:
    """Every strict draft, plus the strongest weaker ones until each group (reactions with
    ``both``, game) has ``minimum`` pooled across the sources."""
    keep: set[tuple[str, int]] = set()
    for group in (REACTION_KINDS, ("game",)):
        items = [
            (key, i, m)
            for key, ms in per_source.items()
            for i, m in enumerate(ms)
            if m["kind"] in group
        ]
        strict = [(k, i) for k, i, m in items if m["strict"]]
        keep.update(strict)
        weak = sorted((x for x in items if not x[2]["strict"]), key=lambda x: -x[2]["strength"])
        keep.update((k, i) for k, i, _ in weak[: max(0, minimum - len(strict))])
    return {
        key: [
            {"t": round(m["t"], 1), "kind": m["kind"], "note": m["note"], "draft": True}
            for i, m in enumerate(ms)
            if (key, i) in keep
        ]
        for key, ms in per_source.items()
    }


def write_drafts(path: Path, source: Path, moments: list[dict[str, Any]], force: bool) -> bool:
    """Write a draft label file; an existing one only with ``force``. Returns whether written."""
    if path.exists() and not force:
        logger.warning("%s exists; not overwritten (use --force)", path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source": source.name,
        "labeller": "draft: scripts/loudness_eval.py --draft-labels (correct me)",
        "draft": True,
        "moments": moments,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


# --------------------------------------------------------------------------- output
def _ratio(m: dict[str, Any], a: str, b: str) -> str:
    return f"{m[a]}/{m[b]}"


def _w(header: str, width: int) -> int:
    """A column's width: ``width``, or wider so the header keeps a space before it."""
    return max(width, len(header) + 1)


def print_clip(clip: dict[str, Any], tops: list[str]) -> None:
    """One table per clip: a row per variant and stage."""
    lab = clip["labels"]
    tr = clip.get("tracks") or {}
    print(
        f"\n{clip['source']}: {lab['reactions']} reaction, {lab['games']} game labels"
        + (f"; tracks {tr}" if tr else "")
    )
    head = f"{'variant':<14}{'stage':<6}"
    widths = {k: (_w("R@" + k, 14), _w("FP@" + k, 14), _w("G@" + k, 10)) for k in tops}
    for k in tops:
        wr, wf, wg = widths[k]
        head += f"{'R@' + k:>{wr}}{'FP@' + k:>{wf}}{'mix':>5}{'G@' + k:>{wg}}{'+/-':>7}"
    print(head + f"{'loud%':>12}  note")
    for name, stages in clip["variants"].items():
        for stage, r in stages.items():
            row = f"{name:<14}{stage:<6}"
            for k in tops:
                m, (wr, wf, wg) = r["at"][k], widths[k]
                rk = _ratio(m, "hit", "reactions") + (f" ({m['n']})" if k == SUGGESTED else "")
                row += f"{rk:>{wr}}{_ratio(m, 'fp', 'n'):>{wf}}{m['fp_mixed']:>5}"
                row += f"{_ratio(m, 'game_reached', 'games'):>{wg}}"
                row += f"{'+' + str(m.get('gained', 0)) + '/-' + str(m.get('lost', 0)):>7}"
            ls = r["loud_share"]
            share = "-" if ls["mean"] is None else f"{ls['mean']:.2f}/{ls['max']:.2f}"
            note = f"n/a, {r['na']}" if r.get("na") else ""
            if "reweighted" in r:
                note = f"{r['reweighted']} reweighted"
            print(row + f"{share:>12}  {note}")


def print_pooled(pooled: dict[str, Any], tops: list[str]) -> None:
    """The pooled table and the criteria."""
    print("\nPooled")
    head = f"{'variant':<14}{'stage':<6}{'recs':>5}"
    widths = {k: (_w("R@" + k, 12), _w("FP@" + k, 12)) for k in tops}
    for k in tops:
        wr, wf = widths[k]
        head += f"{'R@' + k:>{wr}}{'FP@' + k:>{wf}}{'mix':>5}{'+/-':>8}{'maxlost':>8}"
    print(head + "  criteria")
    for name, stages in pooled.items():
        for stage, p in stages.items():
            row = f"{name:<14}{stage:<6}{p['recordings']:>5}"
            for k in tops:
                m, (wr, wf) = p["at"][k], widths[k]
                row += f"{_ratio(m, 'hit', 'reactions'):>{wr}}{_ratio(m, 'fp', 'slots'):>{wf}}"
                row += f"{m['fp_mixed']:>5}"
                row += f"{'+' + str(m['gained']) + '/-' + str(m['lost']):>8}{m['max_lost']:>8}"
            c = p.get("criteria")
            if c:
                drop = "-" if c["fp_drop"] is None else f"{100 * c['fp_drop']:.0f}%"
                row += f"  FP drop {drop}, net {c['net']:+d}: {'PASS' if c['pass'] else 'fail'}"
            print(row)


# --------------------------------------------------------------------------- main
def labels_path(src: Path, labels_dir: Path | None) -> Path:
    """``<labels-dir>/<stem>.json`` (default ``labels/`` next to the source)."""
    return (labels_dir or src.parent / "labels") / f"{src.stem}.json"


def label_dir_writable(folder: Path) -> bool:
    """Whether label files can be written in ``folder``: a probe file in it, or in the nearest
    existing parent when it does not exist yet (so no folder is created for the check). The
    ``os.path`` checks report False where ``Path`` raises on a folder that cannot be opened."""
    probe = folder
    while not os.path.exists(probe):  # noqa: PTH110 - False, not raising, on OSError
        if probe.parent == probe:
            return False
        probe = probe.parent
    if not os.path.isdir(probe):  # noqa: PTH112 - False, not raising, on OSError
        return False
    # One exclusive create: tempfile retries random names on a PermissionError (an ACL that
    # denies file creation on Windows) and can hang for minutes.
    probe_file = Path(probe) / f".loudness-eval-probe-{os.getpid()}"
    try:
        fd = os.open(probe_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return False
    try:
        os.close(fd)
    finally:
        try:
            probe_file.unlink()
        except OSError:
            logger.warning("Could not remove the probe file %s; delete it by hand", probe_file)
    return True


def shared_label_file(src: Path, path: Path, seen: dict[str, Path]) -> bool:
    """Whether an earlier source already uses ``path`` (same stem, also in another case on a
    case-insensitive file system); warns and names both. ``seen`` is keyed by the normalized
    path."""
    key = os.path.normcase(os.path.abspath(path))  # noqa: PTH100 - normalises '..'
    other = seen.get(key)
    if other is not None:
        logger.warning("%s and %s share the label file %s; %s skipped", other, src, path, src)
        return True
    seen[key] = src
    return False


def analyse(src: Path, style: Style, tr: Any, args: argparse.Namespace) -> SourceResult:
    """The source's signals from Cutpoint's caches, no candidates picked. Its profanity gate
    uses the baseline loudness, so the voice variants are not re-gated."""
    return analyze.analyze_source(
        src, style, tr, not args.no_scenes and style.use_scenes, args.max_candidates, {}, pick=False
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="+", type=Path, help="source videos")
    ap.add_argument("--labels-dir", type=Path, help="label files (default: labels/ by each source)")
    ap.add_argument("--style", default="gaming", help="style (default: gaming)")
    ap.add_argument(
        "--variant",
        action="append",
        metavar="SPEC",
        help="baseline, voice, boom:K or reweight:W (repeatable; default: the plan's sweep)",
    )
    ap.add_argument(
        "--with-events",
        action="append",
        metavar="PROJECT",
        default=[],
        help="project whose events.json adds the post-events stage (repeatable)",
    )
    ap.add_argument(
        "--top",
        action="append",
        metavar="N",
        help="top N, or 'suggested' (default: 5 10 suggested)",
    )
    ap.add_argument("--out", type=Path, help="write the results as JSON here")
    ap.add_argument("--cache-dir", type=Path, help="script cache (default: <cache>/loudness_eval)")
    ap.add_argument("--model", help="whisper model the analysis used")
    ap.add_argument("--language", help="transcription language the analysis forced")
    ap.add_argument("--no-scenes", action="store_true", help="as analyze --no-scenes")
    ap.add_argument("--max-candidates", type=int, help="as analyze --max-candidates")
    ap.add_argument("--draft-labels", action="store_true", help="write DRAFT label files")
    ap.add_argument("--force", action="store_true", help="--draft-labels: overwrite label files")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        variants = parse_variants(
            args.variant if args.variant is not None else list(DEFAULT_VARIANTS)
        )
        tops = parse_tops(args.top or list(DEFAULT_TOP))
        style = get_style(args.style)
    except ValueError as err:
        ap.error(str(err))
    if style.base != "gaming":
        logger.warning("#512 measures gaming only; %s is %s", style.name, style.base)
    for s in args.sources:
        if not s.is_file():
            ap.error(f"{s}: not a file")
    if args.draft_labels:
        targets = (labels_path(s, args.labels_dir) for s in args.sources)
        folders = {p.parent for p in targets if args.force or not os.path.exists(p)}  # noqa: PTH110
        for folder in sorted(folders):
            if not label_dir_writable(folder):
                logger.error("Cannot write label files in %s; pass a writable --labels-dir", folder)
                return 1
    root = eval_cache(args.cache_dir)
    from cutpoint.transcribe import Transcriber  # noqa: PLC0415

    tr = Transcriber(model=args.model, language=args.language)
    no_panns = panns_missing()
    booms = BoomGrids(root)
    try:
        if args.draft_labels:
            return draft_main(args, style, tr, booms, no_panns)
        return eval_main(args, style, tr, variants, tops, root, booms, no_panns)
    finally:
        booms.close()
        tr.close()


def draft_main(
    args: argparse.Namespace, style: Style, tr: Any, booms: BoomGrids, no_panns: str | None
) -> int:
    """``--draft-labels``: one DRAFT label file per source."""
    if no_panns:
        logger.warning("Draft game moments without boom peaks: %s", no_panns)
    found: dict[str, list[dict[str, Any]]] = {}
    by_key: dict[str, Path] = {}
    claimed: dict[str, Path] = {}
    for src in args.sources:
        path = labels_path(src, args.labels_dir)
        if shared_label_file(src, path, claimed):
            continue
        if path.exists() and not args.force:
            logger.warning("%s exists; not overwritten (use --force)", path)
            continue
        res = analyse(src, style, tr, args)
        ps = res.signals
        assert ps is not None
        boom = None if no_panns else booms.get(src, ps.n)
        found[str(path)] = draft_moments(ps, boom)
        by_key[str(path)] = src
    chosen = select_drafts(found)
    totals = dict.fromkeys(("reaction", "both", "game"), 0)
    for key, moments in chosen.items():
        if write_drafts(Path(key), by_key[key], moments, args.force):
            for m in moments:
                totals[m["kind"]] += 1
            logger.info("Wrote %d draft moments to %s", len(moments), key)
    print(
        f"Draft labels: {totals['reaction']} reaction, {totals['both']} both, "
        f"{totals['game']} game in {len(chosen)} file(s)"
    )
    if totals["reaction"] + totals["both"] < MIN_DRAFTS or totals["game"] < MIN_DRAFTS:
        logger.warning("Fewer than %d drafts in a group: more footage is needed", MIN_DRAFTS)
    return 0


def eval_main(
    args: argparse.Namespace,
    style: Style,
    tr: Any,
    variants: list[Variant],
    tops: list[str],
    root: Path,
    booms: BoomGrids,
    no_panns: str | None,
) -> int:
    """The evaluation: tables on stdout, JSON in ``--out``."""
    skipped: list[str] = []
    if no_panns and any(v.kind == "boom" for v in variants):
        msg = f"boom variants skipped: {no_panns}"
        logger.warning("%s", msg)
        skipped.append(msg)
        variants = [v for v in variants if v.kind != "boom"]
    loaded = []
    for p in args.with_events:
        project = resolve_project_path(p)
        ev = cache.load_json(project / "events.json")
        if ev is None:
            logger.error("%s: no events.json", project)
            return 1
        loaded.append((project, ev))
    if not loaded and any(v.kind == "reweight" for v in variants):
        msg = "reweight variants skipped: they need --with-events"
        logger.warning("%s", msg)
        skipped.append(msg)
        variants = [v for v in variants if v.kind != "reweight"]
    settings = {
        "language": args.language,
        "whisper_model": tr.model_name,
        "max_candidates": args.max_candidates,
    }
    clips = []
    used: dict[str, Path] = {}
    for src in args.sources:
        lpath = labels_path(src, args.labels_dir)
        if shared_label_file(src, lpath, used):
            continue
        labels = load_labels(lpath, src)
        if labels is None:
            logger.warning("%s: no label file %s; skipped", src.name, lpath)
            continue
        res = analyse(src, style, tr, args)
        ps = res.signals
        assert ps is not None
        folder = source_cache(root, src)
        voice = None
        if any(v.kind.startswith("voice") for v in variants):
            voice = voice_inputs(res, folder)
            if isinstance(voice, str):
                logger.info("%s: voice variants n/a (%s)", src.name, voice)
        boom = booms.get(src, ps.n) if any(v.kind == "boom" for v in variants) else None
        ev = events_for(src, loaded)
        if loaded and ev is None:
            logger.warning("%s: in no --with-events project; pre stage only", src.name)
        result = run_source(src, labels, res, style, variants, tops, ev, settings, voice, boom)
        clips.append(
            {
                "source": src.name,
                "path": str(src),
                "labels": {
                    "reactions": sum(1 for m in labels if m["kind"] in REACTION_KINDS),
                    "games": sum(1 for m in labels if m["kind"] == "game"),
                },
                "tracks": voice[2] if isinstance(voice, tuple) else voice,
                "events": str(ev[0]) if ev else None,
                "variants": result,
            }
        )
        print_clip(clips[-1], tops)
    if not clips:
        logger.error("No source has a label file")
        return 1
    pooled = pool(clips, [v.name for v in variants], tops)
    print_pooled(pooled, tops)
    for msg in skipped:
        print(f"Note: {msg}")
    if args.out:
        data = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "style": style.name,
            "variants": [v.name for v in variants],
            "tops": tops,
            "skipped": skipped,
            "clips": clips,
            "pooled": pooled,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
