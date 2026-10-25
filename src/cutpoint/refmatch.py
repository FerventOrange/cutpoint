"""``cutpoint refmatch``: find a reference edit's shots in the sources and learn from them.

1. **Thumbnails.** The reference and every project source are decoded once to
   64x36 grey frames at 10 fps (NVDEC when possible), cached per file as
   ``cache/<file>/thumbs64x36-10fps.npy``.
2. **Shots.** Cuts in the reference are frames whose correlation with the
   previous frame drops below ``cut_corr``; shots shorter than 0.4 s are merged.
3. **Matching.** Every reference frame is correlated (zero-mean, unit-norm)
   with every source frame. Per shot and source, the per-frame best matches
   vote for a time mapping ``t_src = a + b * t_ref`` (``b`` = 1, or a slow /
   fast-motion factor). The aligned mean correlation is the shot's score; a
   shot is matched when it reaches ``min_score``. When the two best sources
   (or positions) score within ``tie_margin``, the 8 kHz audio decides
   (normalised cross-correlation), since the edit's music makes audio too
   weak to match on its own.
4. **Labels.** Matched source spans are positives. Unmatched parts of used
   clips (with a 1 s guard) and whole unused clips are negatives. Everything
   goes to ``<project>/refmatch.json``; ``coverage`` is the share of the
   edit's duration that was matched.
5. **Fitting** (ski projects): ``fit_weights`` fits non-negative logistic
   regression weights for the ski features on 1 s bins of unmasked footage,
   reports per-feature rank correlation / AUC, evaluates precision and recall
   at the edit's duration before and after (in-sample and grouped by source
   with cross-validation), and saves a learned preset to
   ``<CUTPOINT_HOME>/styles/<name>.json`` (usable as ``--style <name>``).
   ``--taste`` adds a logistic-regression head on the visual layer's CLIP
   frame embeddings, cross-fitted so no score is ever produced by a head
   that saw the same clip. :func:`run_fit` (``cutpoint fit``) fits one preset on
   several projects at once and adds cross-project numbers and ablations.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, paths, state
from cutpoint.config import path_key, source_path
from cutpoint.ffmpeg_utils import FFmpegError, decode_pcm, decode_raw, output_dir, probe, run

logger = logging.getLogger(__name__)

W, H, FPS = 64, 36, 10
AUDIO_SR = 8000
THUMBS_FILE = f"thumbs{W}x{H}-{FPS}fps.npy"
SLOPES = (1.0, 0.5, 0.25, 1 / 3, 2 / 3, 0.75, 1.5, 2.0)


@dataclass(frozen=True)
class MatchParams:
    """Shot detection and matching thresholds."""

    cut_corr: float = 0.6
    min_shot_s: float = 0.4
    min_score: float = 0.80
    margin_score: float = 0.60  # ... or at least this with a clear lead:
    margin: float = 0.25  # over the best other source / position
    split_s: float = 3.0  # shots longer than this are matched in ~2 s pieces
    tie_margin: float = 0.03
    flat_std: float = 4.0  # frames with less grey-level spread are unmatchable
    guard_s: float = 1.0  # around positives, excluded from negatives


# --------------------------------------------------------------------------- decoding
def _decode_thumbs(path: Path, hwaccel: str | None, fallback_cpu: bool = True) -> np.ndarray:
    """One thumbnail decode (CUDA first unless ``hwaccel`` is None)."""
    vf = f"fps={FPS},scale={W}:{H}:flags=area,format=gray"
    data = bytearray()  # one growing buffer: no per-frame arrays to stack
    for _, f in decode_raw(
        path, size=(W, H), vf=vf, pix_fmt="gray", hwaccel=hwaccel, fallback_cpu=fallback_cpu
    ):
        data += f.data
    if not data:
        raise FFmpegError(f"ffmpeg decoded no frames from {path}")
    return np.frombuffer(data, np.uint8).reshape(-1, W * H)


def thumbnails(path: Path) -> np.ndarray:
    """``(N, H*W)`` uint8 grey frames at ``FPS`` (cached per file).

    Decodes on CUDA; if that fails or is clearly shorter than the video (more
    than 1 s missing, e.g. an NVDEC error part-way) it is redone on the CPU and
    the longer result kept. A result that is still short is returned but not
    cached, so the next run tries again.
    """
    cpath = cache.source_dir(path) / THUMBS_FILE
    if cpath.exists():
        return np.load(cpath)
    t0 = time.perf_counter()
    try:
        expected = probe(path).vduration * FPS
    except FFmpegError:
        expected = 0.0  # unknown: accept what decodes
    frames: np.ndarray | None
    try:
        frames = _decode_thumbs(path, "cuda", fallback_cpu=False)
    except FFmpegError as err:
        logger.debug("Thumbnails %s: CUDA decode failed: %s", path.name, err)
        frames = None
    if frames is None or len(frames) < expected - FPS:
        if frames is not None:
            logger.info(
                "Thumbnails %s: %d of ~%.0f frames; retrying on the CPU",
                path.name,
                len(frames),
                expected,
            )
        try:
            cpu = _decode_thumbs(path, None)
        except FFmpegError as err:
            if frames is None:
                raise FFmpegError(f"Could not decode thumbnails of {path}: {err}") from err
            logger.warning("Thumbnails %s: CPU retry failed: %s", path.name, err)
        else:
            frames = cpu if frames is None or len(cpu) >= len(frames) else frames
    if len(frames) < expected - FPS:
        logger.warning(
            "Thumbnails %s: only %d of ~%.0f frames decoded; not caching them",
            path.name,
            len(frames),
            expected,
        )
        return frames
    np.save(cpath, frames)
    logger.info(
        "Thumbnails %s: %d frames in %.1fs", path.name, len(frames), time.perf_counter() - t0
    )
    return frames


# Cache folders (``cache.source_dir``: path + size + mtime) whose audio decode failed in
# this process: not retried or re-logged until the file changes or ``reset_state``.
_AUDIO_FAILED: set[str] = state.register_container(__name__, "_AUDIO_FAILED", set())


def audio_8k(path: Path) -> np.ndarray:
    """Mono 8 kHz float32 audio (cached per file).

    A source without audio streams caches the empty result for good; other
    decode failures are remembered for this process only, so a later run retries.
    """
    cdir = cache.source_dir(path)
    cpath = cdir / "audio8k.npy"
    if cpath.exists():
        return np.load(cpath)
    if str(cdir) in _AUDIO_FAILED:
        return np.zeros(0, np.float32)
    try:
        x = decode_pcm(path, sr=AUDIO_SR).astype(np.float32) / 32768
    except FFmpegError as err:
        x = np.zeros(0, np.float32)
        try:
            no_audio = probe(path).audio_streams == 0
        except FFmpegError:
            no_audio = False
        if no_audio:  # permanent: no audio evidence for this source
            logger.info("%s has no audio stream; audio tie-breaks skipped", path.name)
            np.save(cpath, x)
        else:
            logger.warning("%s", err)
            _AUDIO_FAILED.add(str(cdir))
        return x
    np.save(cpath, x)
    return x


def _normalize(frames: np.ndarray, flat_std: float) -> tuple[np.ndarray, np.ndarray]:
    x = frames.astype(np.float32)
    std = x.std(1)
    x -= x.mean(1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
    return x, std >= flat_std


# --------------------------------------------------------------------------- shots
def detect_shots(ref: np.ndarray, p: MatchParams) -> list[tuple[int, int]]:
    """``[start, end)`` frame ranges of the reference's shots."""
    x, _ = _normalize(ref, 0.0)
    c = (x[1:] * x[:-1]).sum(1)
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(c < p.cut_corr)]
    min_len = round(p.min_shot_s * FPS)
    merged = [0]
    for k in cuts[1:]:
        if k - merged[-1] >= min_len:
            merged.append(k)
    merged.append(len(ref))
    return [(a, b) for a, b in pairwise(merged) if b > a]


# --------------------------------------------------------------------------- matching
def _fit_mapping(t_ref: np.ndarray, t_src: np.ndarray, w: np.ndarray) -> tuple[float, float, int]:
    """Best ``(a, b, inliers)`` for ``t_src = a + b * t_ref`` over candidate slopes."""
    best = (0.0, 1.0, -1)
    for b in SLOPES:
        a_i = t_src - b * t_ref
        a = float(np.median(a_i))
        inl = int((np.abs(a_i - a) < 0.25).sum())
        if inl > best[2] or (inl == best[2] and b == 1.0):
            best = (a, b, inl)
    a, b, _ = best
    inl_mask = np.abs(t_src - b * t_ref - a) < 0.25
    if inl_mask.any():  # refine the intercept on inliers
        a = float(np.average(t_src[inl_mask] - b * t_ref[inl_mask], weights=w[inl_mask] + 1e-6))
    return a, b, int(inl_mask.sum())


def _aligned_score(
    r: np.ndarray, s: np.ndarray, frames: np.ndarray, a: float, b: float, valid: np.ndarray
) -> float:
    t_ref = frames / FPS
    idx = np.round((a + b * t_ref) * FPS).astype(int)
    ok = (idx >= 0) & (idx < len(s)) & valid
    if ok.sum() < max(2, 0.5 * len(frames)):
        return 0.0
    return float((r[frames[ok]] * s[idx[ok]]).sum(1).mean())


def _audio_ncc(
    ref_a: np.ndarray, src_a: np.ndarray, r0: float, r1: float, s0: float
) -> float | None:
    """NCC of the reference audio [r0, r1] against the source at s0 (+-0.3 s search)."""
    n = int((r1 - r0) * AUDIO_SR)
    if n < AUDIO_SR // 2:
        return None
    seg = ref_a[int(r0 * AUDIO_SR) : int(r0 * AUDIO_SR) + n]
    lo = max(0, int((s0 - 0.3) * AUDIO_SR))
    win = src_a[lo : lo + n + int(0.6 * AUDIO_SR)]
    if len(seg) < n or len(win) <= n:
        return None
    seg = (seg - seg.mean()) / (seg.std() + 1e-6)
    v = np.lib.stride_tricks.sliding_window_view(win, n)[::40]
    v = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-6)
    return float((v @ seg).max() / n)


def match_reference(reference: Path, sources: list[Path], p: MatchParams) -> dict[str, Any]:
    """Match every shot of ``reference`` to a source span (see module docs)."""
    t0 = time.perf_counter()
    ref_raw = thumbnails(reference)
    r, r_valid = _normalize(ref_raw, p.flat_std)
    shots = detect_shots(ref_raw, p)
    logger.info("Reference %s: %.1f s, %d shots", reference.name, len(r) / FPS, len(shots))
    src_mats: list[tuple[np.ndarray, np.ndarray]] = []
    best_s = np.full((len(r), len(sources)), -1.0, np.float32)
    best_i = np.zeros((len(r), len(sources)), np.int32)
    for k, src in enumerate(sources):
        s, s_valid = _normalize(thumbnails(src), p.flat_std)
        src_mats.append((s, s_valid))
        for j0 in range(0, len(s), 8192):
            blk = r @ s[j0 : j0 + 8192].T
            blk[:, ~s_valid[j0 : j0 + 8192]] = -1.0
            i = blk.argmax(1)
            v = blk[np.arange(len(r)), i]
            better = v > best_s[:, k]
            best_s[better, k] = v[better]
            best_i[better, k] = i[better] + j0
    thumbs_s = time.perf_counter() - t0
    ref_audio = None
    out_shots = []
    pieces: list[tuple[int, int, int, int]] = []  # (shot, part, f0, f1)
    for n_shot, (f0, f1) in enumerate(shots):
        if (f1 - f0) / FPS > p.split_s:  # a dissolve can hide a cut: match in pieces
            k = max(2, round((f1 - f0) / FPS / 2.0))
            edges = np.linspace(f0, f1, k + 1).round().astype(int)
            pieces += [(n_shot, j, int(a), int(b)) for j, (a, b) in enumerate(pairwise(edges))]
        else:
            pieces.append((n_shot, 0, f0, f1))
    for n_shot, part, f0, f1 in pieces:
        frames = np.arange(f0, f1)
        valid = r_valid[frames]
        rec: dict[str, Any] = {
            "shot": n_shot,
            "part": part,
            "ref_start": round(f0 / FPS, 2),
            "ref_end": round(f1 / FPS, 2),
            "matched": False,
        }
        if valid.mean() < 0.5:
            rec["reason"] = "flat frames (black, white or title)"
            out_shots.append(rec)
            continue
        cands = []
        for k in range(len(sources)):
            fr = frames[valid]
            a, b, inl = _fit_mapping(fr / FPS, best_i[fr, k] / FPS, best_s[fr, k])
            s, s_valid = src_mats[k]
            cands.append((_aligned_score(r, s, frames, a, b, valid), k, a, b, inl))
        cands.sort(reverse=True)
        score, k, a, b, inl = cands[0]
        second = cands[1][0] if len(cands) > 1 else 0.0
        method = "visual"
        if len(cands) > 1 and score - second < p.tie_margin and score >= p.min_score:
            if ref_audio is None:
                ref_audio = audio_8k(reference)
            nccs = []
            for _sc, kk, aa, bb, _ in cands[:2]:
                ncc = (
                    _audio_ncc(
                        ref_audio, audio_8k(sources[kk]), f0 / FPS, f1 / FPS, aa + bb * f0 / FPS
                    )
                    if bb == 1.0
                    else None
                )
                nccs.append(ncc if ncc is not None else -1.0)
            rec["audio_ncc"] = [round(x, 3) for x in nccs]
            if nccs[1] > nccs[0] + 0.05:
                score, k, a, b, inl = cands[1]
                second = cands[0][0]
            method = "visual+audio"
        rec.update(
            {
                "score": round(score, 3),
                "second_score": round(second, 3),
                "source": str(sources[k]),
                "speed": round(1 / b, 2) if b else None,
                "src_start": round(a + b * f0 / FPS, 2),
                "src_end": round(a + b * f1 / FPS, 2),
                "inliers": inl,
                "method": method,
            }
        )
        if score >= p.min_score:
            rec["matched"] = True
        elif score >= p.margin_score and score - second >= p.margin:
            rec["matched"] = True
            rec["method"] += " (margin)"
        else:
            rec["reason"] = f"best score {score:.2f} < {p.min_score} (lead {score - second:.2f})"
        out_shots.append(rec)
    dur = len(r) / FPS
    matched = [s for s in out_shots if s["matched"]]
    cov = sum(s["ref_end"] - s["ref_start"] for s in matched) / dur if dur else 0.0
    positives = _merge_spans(
        [
            (s["source"], min(s["src_start"], s["src_end"]), max(s["src_start"], s["src_end"]))
            for s in matched
        ]
    )
    durations = {str(s): probe(s).duration for s in sources}
    used = sorted({s for s, _, _ in positives})
    negatives = []
    for src in map(str, sources):
        spans = [(a, b) for s, a, b in positives if s == src]
        cur = 0.0
        for a, b in spans:
            if a - p.guard_s > cur:
                negatives.append((src, round(cur, 2), round(a - p.guard_s, 2)))
            cur = max(cur, b + p.guard_s)
        if durations[src] > cur:
            negatives.append((src, round(cur, 2), round(durations[src], 2)))
    return {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "reference": str(reference),
        "reference_duration": round(dur, 2),
        "params": asdict(p),
        "shots": out_shots,
        "n_shots": len(shots),
        "n_pieces": len(out_shots),
        "n_matched": len({s["shot"] for s in matched}),
        "coverage": round(cov, 4),
        "positives": [
            {"source": s, "start": round(a, 2), "end": round(b, 2)} for s, a, b in positives
        ],
        "negatives": [{"source": s, "start": a, "end": b} for s, a, b in negatives],
        "used_sources": used,
        "unused_sources": [str(s) for s in sources if str(s) not in used],
        "timings": {
            "thumbnails_and_correlation_s": round(thumbs_s, 1),
            "total_s": round(time.perf_counter() - t0, 1),
        },
    }


def _merge_spans(spans: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for s, a, b in sorted(spans):
        if out and out[-1][0] == s and a <= out[-1][2] + 0.2:
            out[-1] = (s, out[-1][1], max(out[-1][2], b))
        else:
            out.append((s, a, b))
    return out


def check_frames(result: dict[str, Any], out_dir: Path, n: int = 6) -> list[Path]:
    """Side-by-side JPEGs (reference | matched source) for ``n`` spread-out matches."""
    with output_dir(out_dir):  # interrupted: no empty folder left behind
        matched = [s for s in result["shots"] if s["matched"]]
        picks = (
            [matched[i] for i in np.linspace(0, len(matched) - 1, min(n, len(matched)), dtype=int)]
            if matched
            else []
        )
        frames = []
        for s in picks:
            tr = (s["ref_start"] + s["ref_end"]) / 2
            ts = s["src_start"] + (tr - s["ref_start"]) / (s.get("speed") or 1)
            out = (
                out_dir / f"shot{s['shot']:03d}{chr(97 + s.get('part', 0))}_"
                f"{source_path(s['source']).stem}_{ts:.1f}s.jpg"
            )
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{tr:.2f}",
                "-i",
                result["reference"],
                "-ss",
                f"{ts:.2f}",
                "-i",
                s["source"],
                "-filter_complex",
                "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack",
                "-frames:v",
                "1",
                str(out),
            ]
            with contextlib.suppress(FFmpegError):
                run(cmd)
            if out.exists():
                frames.append(out)
        return frames


# --------------------------------------------------------------------------- fitting (ski)
BIN_S = 1.0
TASTE_C = 0.5  # inverse L2 strength of the taste head (unit-norm CLIP embeddings)
Key = tuple[int, int]  # (dataset index, index into that dataset's saved features)


@dataclass
class FitData:
    """One project's saved ski feature grids plus its refmatch labels, loaded once."""

    project: Path
    result: dict[str, Any]
    feats: list[dict[str, Any]]

    @property
    def name(self) -> str:
        """Project folder name (for reports)."""
        return self.project.name


def load_fit_data(project: Path, result: dict[str, Any] | None = None) -> FitData:
    """Load ``refmatch.json`` (unless given) and the project's ``ski_features``.

    Raises:
        FileNotFoundError: If the project has no refmatch.json / ski features.
    """
    from cutpoint import ski  # noqa: PLC0415

    result = result or cache.load_json(project / "refmatch.json")
    if not result:
        raise FileNotFoundError(
            f"{project / 'refmatch.json'} not found; run `cutpoint refmatch` on this project first"
        )
    return FitData(project, result, ski.load_project_features(project))


def _positives(result: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    pos: dict[str, list[tuple[float, float]]] = {}
    for sp in result["positives"]:
        pos.setdefault(path_key(str(Path(sp["source"]))), []).append((sp["start"], sp["end"]))
    return pos


def _bins(
    datasets: list[FitData],
    keys: list[Key],
    fmap: dict[Key, dict[str, Any]],
    weights_keys: list[str],
    groups: dict[Key, int],
    friends_strong: float = 7.0,
) -> dict[str, Any]:
    """1 s bins of the given saved sources: features, eligibility, labels, groups."""
    from cutpoint import ski  # noqa: PLC0415

    X, y, elig, grp, srcs, friends_b, rawmask_b = [], [], [], [], [], [], []
    per = int(BIN_S * 10)
    pos_of = [_positives(ds.result) for ds in datasets]
    for k in keys:
        d = fmap[k]
        n = len(d["mask"])
        nb = n // per
        if nb == 0:
            continue
        lab = np.zeros(n)
        for a, b in pos_of[k[0]].get(path_key(str(Path(d["source"]))), []):
            lab[max(0, int((a - d["offset"]) * 10)) : max(0, int((b - d["offset"]) * 10))] = 1
        em = ski.effective_mask(d["mask"], d["friends"], friends_strong)
        friends_b.append(d["friends"][: nb * per].reshape(nb, per).max(1))
        rawmask_b.append(d["mask"][: nb * per].reshape(nb, per).mean(1) >= 0.5)
        X.append(
            np.stack(
                [
                    d["features"].get(f, np.zeros(n))[: nb * per].reshape(nb, per).mean(1)
                    for f in weights_keys
                ],
                1,
            )
        )
        y.append(lab[: nb * per].reshape(nb, per).mean(1) >= 0.5)
        elig.append(em[: nb * per].reshape(nb, per).mean(1) < 0.5)
        grp.append(np.full(nb, groups[k]))
        srcs += [d["source"]] * nb
    return {
        "X": np.concatenate(X),
        "y": np.concatenate(y),
        "elig": np.concatenate(elig),
        "group": np.concatenate(grp),
        "source": srcs,
        "friends": np.concatenate(friends_b),
        "rawmask": np.concatenate(rawmask_b),
    }


def logistic_nonneg(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 4000, lr: float = 0.5
) -> tuple[np.ndarray, float]:
    """Class-balanced logistic regression with non-negative weights (projected GD)."""
    _n, d = X.shape
    w, b = np.zeros(d), 0.0
    pw = 0.5 / max(1, y.sum())
    nw = 0.5 / max(1, (~y).sum())
    sw = np.where(y, pw, nw)
    for _ in range(iters):
        z = X @ w + b
        pr = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = sw * (pr - y)
        w = np.maximum(0.0, w - lr * (X.T @ g + l2 * w))
        b -= lr * g.sum()
    return w, b


def _rankdata(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (like scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    """ROC AUC via ranks (ties averaged)."""
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    ranks = _rankdata(score)
    npos = y.sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * (~y).sum()))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation (None for constant inputs)."""
    if np.ptp(x) == 0 or np.ptp(y.astype(float)) == 0:
        return None
    return float(np.corrcoef(_rankdata(x), _rankdata(y.astype(float)))[0, 1])


def evaluate(
    project: Path,
    result: dict[str, Any],
    weights: dict[str, float],
    only_sources: set[str] | None = None,
    budget: float | None = None,
    friends_strong: float = 7.0,
    feats: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank auto-picked clips with ``weights``; precision/recall at the edit's duration.

    Clips are picked per source exactly as ``analyze`` would (ski geometry and
    masks), ranked by score and taken until their total reaches ``budget``
    (default: the reference duration, scaled to the share of positives in
    ``only_sources``). ``moments`` counts matched reference shots whose
    source span overlaps a selected clip. ``feats`` replaces the saved
    feature grids (e.g. with a held-out taste score).
    """
    from cutpoint import ski  # noqa: PLC0415

    picks = []
    for d in feats if feats is not None else ski.load_project_features(project):
        if only_sources is not None and path_key(d["source"]) not in only_sources:
            continue
        evs = d["events"]
        for pk in ski.pick(
            d["features"],
            weights,
            d["mask"],
            d["friends"],
            evs,
            d["duration"],
            10_000,
            d["offset"],
            friends_strong,
        ):
            picks.append(
                (pk.peak, d["source"], pk.start + d["offset"], pk.end + d["offset"], pk.kind)
            )
    picks.sort(key=lambda p: -p[0])
    pos = [
        (path_key(sp["source"]), sp["start"], sp["end"])
        for sp in result["positives"]
        if only_sources is None or path_key(sp["source"]) in only_sources
    ]
    pos_total = sum(b - a for _, a, b in pos)
    if budget is None:
        all_pos = sum(sp["end"] - sp["start"] for sp in result["positives"]) or 1.0
        budget = result["reference_duration"] * pos_total / all_pos
    chosen, total = [], 0.0
    for p in picks:
        if total >= budget:
            break
        chosen.append(p)
        total += p[3] - p[2]

    def overlap(src: str, a: float, b: float) -> float:
        return sum(max(0.0, min(b, pb) - max(a, pa)) for s, pa, pb in pos if s == src)

    ov = sum(overlap(path_key(s), a, b) for _, s, a, b, _ in chosen)
    shots = [
        s
        for s in result["shots"]
        if s["matched"] and (only_sources is None or path_key(s["source"]) in only_sources)
    ]
    hit_ids: set[int] = set()
    for s in shots:  # a shot counts once, whichever of its pieces is hit
        a, b = sorted((s["src_start"], s["src_end"]))
        if any(
            path_key(cs) == path_key(s["source"]) and min(b, cb) > max(a, ca)
            for _, cs, ca, cb, _ in chosen
        ):
            hit_ids.add(s["shot"])
    hit = len(hit_ids)
    shots = list({s["shot"]: s for s in shots}.values())
    kinds: dict[str, int] = {}
    for c in chosen:
        kinds[c[4]] = kinds.get(c[4], 0) + 1
    return {
        "budget_s": round(budget, 1),
        "selected_s": round(total, 1),
        "clips": len(chosen),
        "precision": round(ov / total, 3) if total else 0.0,
        "recall": round(ov / pos_total, 3) if pos_total else 0.0,
        "overlap_s": round(ov, 1),
        "positives_s": round(pos_total, 1),
        "moments_hit": hit,
        "moments": len(shots),
        "moment_recall": round(hit / len(shots), 3) if shots else 0.0,
        "kinds": kinds,
    }


def pool(evs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool evaluations of disjoint source sets (folds, projects) into one P/R."""
    sel = sum(e["selected_s"] for e in evs)
    tp = sum(e.get("overlap_s", e["precision"] * e["selected_s"]) for e in evs)
    pos_tot = sum(
        e.get("positives_s", (e["precision"] * e["selected_s"] / e["recall"]) if e["recall"] else 0)
        for e in evs
    )
    hit, mom = sum(e["moments_hit"] for e in evs), sum(e["moments"] for e in evs)
    return {
        "precision": round(tp / sel, 3) if sel else 0.0,
        "recall": round(tp / pos_tot, 3) if pos_tot else 0.0,
        "moment_recall": round(hit / mom, 3) if mom else 0.0,
        "moments_hit": hit,
        "moments": mom,
        "selected_s": round(sel, 1),
    }


# ---- taste head: logistic regression on CLIP frame embeddings
def _taste_frames(datasets: list[FitData], keys: list[Key]) -> dict[Key, dict[str, np.ndarray]]:
    """Mean-crop CLIP embeddings, part-relative frame times and labels per saved source."""
    from cutpoint.events.visual import load_visual  # noqa: PLC0415

    out: dict[Key, dict[str, np.ndarray]] = {}
    pos_of = [_positives(ds.result) for ds in datasets]
    for k in keys:
        d = datasets[k[0]].feats[k[1]]
        vd = load_visual(Path(d["source"]))
        if vd is None or not len(vd.t):
            continue
        t = vd.t.astype(float) - d["offset"]
        ok = (t >= 0) & (t <= d["duration"])
        y = np.zeros(int(ok.sum()), bool)
        for a, b in pos_of[k[0]].get(path_key(str(Path(d["source"]))), []):
            y |= (t[ok] + d["offset"] >= a) & (t[ok] + d["offset"] < b)
        out[k] = {"emb": vd.mean_emb[ok], "t": t[ok], "y": y}
    return out


def fit_taste(
    frames: dict[Key, dict[str, np.ndarray]], keys: list[Key], c: float = TASTE_C
) -> dict[str, Any] | None:
    """Class-balanced L2 logistic regression on the frames of ``keys`` (None if one class).

    Only clips the editor used are trained on (kept vs not-kept seconds within
    a clip): whole unused clips are left out for many reasons (duplicates, a
    day's quota), and with them in, the head learns which clips rather than
    which moments. On a day of helmet-cam ski footage that raised the clip-held-out AUC from
    0.50 to 0.54; a random frame split reaches 0.86 only through leakage
    between neighbouring frames, which is why every evaluation holds out
    whole clips.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    ks = [k for k in keys if k in frames and frames[k]["y"].any()]
    if not ks:
        return None
    E = np.concatenate([frames[k]["emb"] for k in ks])
    y = np.concatenate([frames[k]["y"] for k in ks])
    if y.sum() < 5 or (~y).sum() < 5:
        return None
    m = LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
    m.fit(E * 10.0, y)  # x10: unit-norm embeddings need a larger scale for L2 to bite
    return {
        "w": (m.coef_[0] * 10.0).astype(np.float32),
        "b": float(m.intercept_[0]),
        "c": c,
        "frames": len(y),
        "positive_frames": int(y.sum()),
    }


def _taste_grid(
    fr: dict[str, np.ndarray] | None, n: int, taste: dict[str, Any] | None
) -> np.ndarray:
    from cutpoint.events.visual import to_grid  # noqa: PLC0415
    from cutpoint.ski import DT, smooth3  # noqa: PLC0415

    if fr is None or taste is None or not len(fr["t"]):
        return np.zeros(n)
    z = fr["emb"] @ taste["w"] + taste["b"]
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return to_grid(fr["t"], smooth3(p), n, DT)


def _with_taste(
    fmap: dict[Key, dict[str, Any]],
    frames: dict[Key, dict[str, np.ndarray]],
    model_of: dict[Key, dict[str, Any] | None],
) -> dict[Key, dict[str, Any]]:
    """Copies of the feature dicts whose ``taste`` grid comes from ``model_of[key]``."""
    out = dict(fmap)
    for k, model in model_of.items():
        d = fmap[k]
        out[k] = {
            **d,
            "features": {
                **d["features"],
                "taste": _taste_grid(frames.get(k), len(d["mask"]), model),
            },
        }
    return out


def _crossfit(
    frames: dict[Key, dict[str, np.ndarray]],
    keys: list[Key],
    groups: dict[Key, int],
    folds: int,
    seed: int,
    c: float,
) -> dict[Key, dict[str, Any] | None]:
    """Out-of-fold taste models: each key gets a model trained without its group."""
    uniq = sorted({groups[k] for k in keys})
    perm = np.random.default_rng(seed).permutation(uniq)
    fold_of = {g: i % folds for i, g in enumerate(perm)}
    out: dict[Key, dict[str, Any] | None] = {}
    for f in range(folds):
        test = [k for k in keys if fold_of[groups[k]] == f]
        if not test:
            continue
        model = fit_taste(frames, [k for k in keys if fold_of[groups[k]] != f], c)
        out.update(dict.fromkeys(test, model))
    return out


# ---- training / evaluation over sets of saved sources
@dataclass
class _Ctx:
    datasets: list[FitData]
    fmap: dict[Key, dict[str, Any]]
    groups: dict[Key, int]
    frames: dict[Key, dict[str, np.ndarray]]
    keys_w: list[str]
    taste: bool
    taste_c: float


def _evaluate_keys(
    ctx: _Ctx,
    keys: list[Key],
    fmap: dict[Key, dict[str, Any]],
    weights: dict[str, float],
    th: float,
) -> dict[str, Any]:
    """Evaluate per project on the sources of ``keys`` and pool the results."""
    evs = []
    for di, ds in enumerate(ctx.datasets):
        ks = [k for k in keys if k[0] == di]
        if not ks:
            continue
        srcs = {path_key(fmap[k]["source"]) for k in ks}
        evs.append(
            evaluate(
                ds.project, ds.result, weights, srcs, friends_strong=th, feats=[fmap[k] for k in ks]
            )
        )
    return {**pool(evs), "per_project": evs}


def _train(
    ctx: _Ctx, keys: list[Key], thresholds: tuple[float, ...], th_default: float, seed: int = 0
) -> dict[str, Any]:
    """Fit taste (cross-fitted) and weights on ``keys``; pick ``mask_friends``."""
    fmap = ctx.fmap
    taste_model = None
    if ctx.taste:
        oof = _crossfit(ctx.frames, keys, ctx.groups, 4, seed + 1, ctx.taste_c)
        fmap = _with_taste(ctx.fmap, ctx.frames, oof)
        taste_model = fit_taste(ctx.frames, keys, ctx.taste_c)
    search = []
    for th in thresholds:
        d = _bins(ctx.datasets, keys, fmap, ctx.keys_w, ctx.groups, th)
        w, b = logistic_nonneg(d["X"][d["elig"]], d["y"][d["elig"]])
        tot = float(w.sum())
        wts = {
            k: round(float(v / tot), 4) if tot > 0 else 0.0
            for k, v in zip(ctx.keys_w, w, strict=True)
        }
        ev = _evaluate_keys(ctx, keys, fmap, wts, th)
        search.append({"mask_friends": th, "weights": wts, "intercept": b, "eval": ev, "data": d})
    best = max(
        search,
        key=lambda s: (
            s["eval"]["recall"] + s["eval"]["precision"],
            -abs(s["mask_friends"] - th_default),
        ),
    )
    return {**best, "search": search, "taste": taste_model, "fmap": fmap}


def mask_modes(thresholds: tuple[float, ...], default: float) -> tuple[float, ...]:
    """``mask_friends`` candidates reduced to the two masks they can give: soft and hard.

    Friends no longer lift the lift/stop mask, so every value above 0 is the
    same hard mask; an older list such as ``(0, 5, 6, 7, 8, 11)`` fits each mask
    once. The hard mask takes ``default`` (the base style's value), or
    :data:`cutpoint.ski.FRIENDS_STRONG` when that is the soft mask.
    """
    from cutpoint.ski import FRIENDS_STRONG  # noqa: PLC0415

    hard = default if default > 0 else float(FRIENDS_STRONG)
    return tuple(dict.fromkeys(0.0 if t <= 0 else hard for t in thresholds))


def fit_weights(
    project: Path | list[FitData],
    result: dict[str, Any] | None = None,
    base: str = "ski",
    name: str | None = None,
    folds: int = 4,
    thresholds: tuple[float, ...] = (0, 7),
    features: list[str] | None = None,
    taste: bool = False,
    taste_c: float = TASTE_C,
    save: bool = True,
    cross: bool = True,
    vlm_friends_only: bool = False,
) -> dict[str, Any]:
    """Fit ski weights to one or more reference edits and save a learned preset.

    For each ``mask_friends`` mode (above 0 = hard lift/stop mask; 0 = soft
    mask, the ``stopped`` feature weight decides; see :func:`mask_modes`) the
    weights are fitted on the bins that mode leaves eligible, and the pair
    with the best in-sample precision + recall at the edits' durations is
    kept. Cross-validation (whole clips held out, across all projects) then
    re-fits the weights for that mode on each training fold. With ``taste`` the taste head is cross-fitted too: weights
    are fitted on out-of-fold taste scores, and held-out clips are scored by a
    head that never saw them. With two or more projects, each project is also
    predicted by a model trained on the others alone (cross-project).

    Args:
        project: A project folder (with ``result``), or loaded :class:`FitData`.
        result: refmatch.json content for a single project.
        base: Built-in preset whose weights / feature names are fitted.
        name: Preset name (``<CUTPOINT_HOME>/styles/<name>.json``).
        folds: Cross-validation folds.
        thresholds: ``mask_friends`` candidates (older numeric lists are
            accepted; values above 0 collapse into one hard-mask fit).
        features: Feature subset (default: every feature of ``base``).
        taste: Fit and use the taste head.
        taste_c: Inverse L2 strength of the taste head.
        save: Write the preset (False for ablations).
        cross: Run the cross-project evaluation.
        vlm_friends_only: Use the VLM friends tags alone as the ``friends``
            grid (for ablations). Friends no longer lift the mask, so this
            only changes the friends numbers of the mask report.

    Returns:
        The fit report (also written into the preset JSON).
    """
    from cutpoint.styles import get_style  # noqa: PLC0415

    datasets = project if isinstance(project, list) else [load_fit_data(project, result)]
    base_style = get_style(base)
    keys_w = [k for k in (features or list(base_style.weights)) if taste or k != "taste"]
    if taste and "taste" not in keys_w:
        keys_w.append("taste")
    default = base_style.weights
    th_default = base_style.mask_friends
    thresholds = mask_modes(thresholds, th_default)
    fmap: dict[Key, dict[str, Any]] = {}
    groups: dict[Key, int] = {}
    gid: dict[tuple[int, str], int] = {}
    for di, ds in enumerate(datasets):
        for fi, d in enumerate(ds.feats):
            if vlm_friends_only and "friends_vlm" in d["features"]:
                d = {**d, "friends": d["features"]["friends_vlm"]}
            fmap[(di, fi)] = d
            groups[(di, fi)] = gid.setdefault((di, path_key(d["source"])), len(gid))
    all_keys = list(fmap)
    frames = _taste_frames(datasets, all_keys) if taste else {}
    ctx = _Ctx(datasets, fmap, groups, frames, keys_w, taste, taste_c)

    final = _train(ctx, all_keys, thresholds, th_default)
    th, learned, data = final["mask_friends"], final["weights"], final["data"]
    for s in final["search"]:
        logger.info(
            "  mask_friends %-4g -> P %.3f R %.3f moments %d/%d",
            s["mask_friends"],
            s["eval"]["precision"],
            s["eval"]["recall"],
            s["eval"]["moments_hit"],
            s["eval"]["moments"],
        )
    X, y, elig = data["X"], data["y"], data["elig"]
    Xe, ye = X[elig], y[elig]
    feat_stats = {}
    for j, k in enumerate(keys_w):
        feat_stats[k] = {
            "spearman": _r(spearman(Xe[:, j], ye)),
            "auc": _r(auc(Xe[:, j], ye)),
            "mean_pos": _r(float(Xe[ye, j].mean()) if ye.any() else None),
            "mean_neg": _r(float(Xe[~ye, j].mean()) if (~ye).any() else None),
        }

    # Grouped cross-validation (hold out whole clips) at the chosen threshold.
    uniq = sorted(set(groups.values()))
    perm = np.random.default_rng(0).permutation(uniq)
    fold_of = {g: i % folds for i, g in enumerate(perm)}
    cv_eval: dict[str, list[dict[str, Any]]] = {"default": [], "learned": []}
    cv_scores, cv_y = [], []
    for f in range(folds):
        test = [k for k in all_keys if fold_of[groups[k]] == f]
        train = [k for k in all_keys if fold_of[groups[k]] != f]
        if not test or not train:
            continue
        tr = _train(ctx, train, (th,), th_default, seed=f + 1)
        if not tr["data"]["y"][tr["data"]["elig"]].any():
            continue
        fm = _with_taste(fmap, frames, dict.fromkeys(test, tr["taste"])) if taste else fmap
        cv_eval["learned"].append(_evaluate_keys(ctx, test, fm, tr["weights"], th))
        cv_eval["default"].append(_evaluate_keys(ctx, test, fmap, default, th_default))
        dt = _bins(datasets, test, fm, keys_w, groups, th)
        e = dt["elig"]
        cv_scores.append(dt["X"][e] @ np.array([tr["weights"][k] for k in keys_w]))
        cv_y.append(dt["y"][e])

    cross_rep = []
    if cross and len(datasets) > 1:
        for a in range(len(datasets)):
            tr_keys = [k for k in all_keys if k[0] == a]
            tr = _train(ctx, tr_keys, thresholds, th_default)
            for b in range(len(datasets)):
                if b == a:
                    continue
                te = [k for k in all_keys if k[0] == b]
                fm = _with_taste(fmap, frames, dict.fromkeys(te, tr["taste"])) if taste else fmap
                cross_rep.append(
                    {
                        "train": datasets[a].name,
                        "test": datasets[b].name,
                        "mask_friends": tr["mask_friends"],
                        "weights": tr["weights"],
                        "learned": _evaluate_keys(ctx, te, fm, tr["weights"], tr["mask_friends"]),
                        "default": _evaluate_keys(ctx, te, fmap, default, th_default),
                    }
                )

    fr, rm = data["friends"], data["rawmask"]
    masked_pos = float(y[~elig].sum() * BIN_S)
    mask_info = {
        "positives_s": float(y.sum() * BIN_S),
        "in_lift_or_stop_s": float((y & rm).sum() * BIN_S),
        "unlocked_by_vlm_friends_s": float((y & rm & elig).sum() * BIN_S),
        "still_masked_s": masked_pos,
        "masked_with_friends_tag_ge5_s": float((y & ~elig & (fr >= 0.5)).sum() * BIN_S),
        "share_of_footage_masked": round(float(rm.mean()), 3),
    }
    dd = _bins(datasets, all_keys, fmap, keys_w, groups, th_default)
    de = dd["elig"]
    report = {
        "mask_friends": th,
        "features_fitted": keys_w,
        "projects": [ds.name for ds in datasets],
        "threshold_search": [
            {
                "mask_friends": s["mask_friends"],
                "precision": s["eval"]["precision"],
                "recall": s["eval"]["recall"],
                "moments_hit": s["eval"]["moments_hit"],
            }
            for s in final["search"]
        ],
        "n_bins": len(y),
        "eligible_bins": int(elig.sum()),
        "positive_bins": int(y.sum()),
        "positive_bins_eligible": int(ye.sum()),
        "positives_in_masked_s": masked_pos,
        "mask": mask_info,
        "features": feat_stats,
        "auc": {
            "default": _r(
                auc(dd["X"][de] @ np.array([default.get(k, 0.0) for k in keys_w]), dd["y"][de])
            ),
            "learned_in_sample": _r(auc(Xe @ np.array([learned[k] for k in keys_w]), ye)),
            "learned_cv": _r(auc(np.concatenate(cv_scores), np.concatenate(cv_y)))
            if cv_scores
            else None,
        },
        "at_edit_duration": {
            "default": _evaluate_keys(ctx, all_keys, fmap, default, th_default),
            "learned_in_sample": final["eval"],
        },
        "cv_at_edit_duration": {
            "default": pool(cv_eval["default"]),
            "learned": pool(cv_eval["learned"]),
        },
        "cross_project": [dict(c) for c in cross_rep],
        "intercept": round(float(final["intercept"]), 3),
    }
    if not save:
        return report
    name = name or f"{base}_learned"
    learned_dir = paths.styles_dir()
    learned_dir.mkdir(parents=True, exist_ok=True)
    preset = {
        "name": name,
        "base": base,
        "weights": learned,
        "mask_friends": th,
        "default_weights": default,
        "created": datetime.now().isoformat(timespec="seconds"),
        "project": str(datasets[0].project),
        "reference": datasets[0].result["reference"],
        "projects": [str(ds.project) for ds in datasets],
        "references": [ds.result["reference"] for ds in datasets],
        "fit": report,
    }
    if taste and final["taste"] is not None:
        tpath = learned_dir / f"{name}.taste.npz"
        np.savez(tpath, w=final["taste"]["w"], b=np.float32(final["taste"]["b"]))
        preset["taste"] = tpath.name
        report["taste_head"] = {k: v for k, v in final["taste"].items() if k not in ("w", "b")}
    path = learned_dir / f"{name}.json"
    cache.save_json(path, preset)
    report["preset"] = str(path)
    return report


ABLATION_SETS = {
    "old features (before; VLM-only friends mask)": "old",
    "old + beauty": ["beauty"],
    "old + people": ["people"],
    "old + group": ["group"],
    "old + laughter": ["laughter"],
    "old + voice": ["voice"],
    "old + taste": ["taste"],
    "all": "all",
    "all - vlm (no VLM tags)": "no_vlm",
    "all - taste": "no_taste",
}


def ablate(
    datasets: list[FitData],
    base: str = "ski",
    sets: dict[str, Any] | None = None,
    thresholds: tuple[float, ...] = (0, 7),
) -> list[dict[str, Any]]:
    """Refit (without saving) on feature subsets; held-out P/R per subset.

    VLM tags enter through ``vlm`` / ``scenic``; "no_vlm" drops those
    features. ``thresholds`` are ``mask_friends`` modes, as for :func:`fit_weights`.
    """
    from cutpoint.ski import FEATURES, NEW_FEATURES  # noqa: PLC0415

    old = [f for f in FEATURES if f not in NEW_FEATURES]
    rows = []
    for label, spec in (sets or ABLATION_SETS).items():
        vlm_only = spec == "old"
        if spec == "old":
            feats, taste = old, False
        elif spec == "all":
            feats, taste = list(FEATURES), True
        elif spec == "no_vlm":
            feats, taste = [f for f in FEATURES if f not in ("vlm", "scenic")], True
        elif spec == "no_taste":
            feats, taste = [f for f in FEATURES if f != "taste"], False
        else:
            feats, taste = old + [f for f in spec if f != "taste"], "taste" in spec
        t0 = time.perf_counter()
        rep = fit_weights(
            datasets,
            base=base,
            features=feats,
            taste=taste,
            save=False,
            thresholds=thresholds,
            cross=len(datasets) > 1,
            vlm_friends_only=vlm_only,
        )
        cv = rep["cv_at_edit_duration"]["learned"]
        row = {
            "set": label,
            "features": feats,
            "taste": taste,
            "mask_friends": rep["mask_friends"],
            "in_sample": {
                k: rep["at_edit_duration"]["learned_in_sample"][k] for k in ("precision", "recall")
            },
            "cv": {k: cv[k] for k in ("precision", "recall", "moment_recall")},
            "auc_cv": rep["auc"]["learned_cv"],
            "cross": [
                {
                    "train": c["train"],
                    "test": c["test"],
                    "precision": c["learned"]["precision"],
                    "recall": c["learned"]["recall"],
                }
                for c in rep["cross_project"]
            ],
            "seconds": round(time.perf_counter() - t0, 1),
        }
        logger.info(
            "ablation %-26s CV P %.3f R %.3f AUC %s | %s (%.0fs)",
            label,
            cv["precision"],
            cv["recall"],
            rep["auc"]["learned_cv"],
            "; ".join(
                f"{c['train']}->{c['test']} P {c['precision']:.3f} R {c['recall']:.3f}"
                for c in row["cross"]
            )
            or "-",
            row["seconds"],
        )
        rows.append(row)
    return rows


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(v, nd)


# --------------------------------------------------------------------------- entry
def run_refmatch(
    project: Path,
    reference: Path,
    *,
    params: MatchParams | None = None,
    fit: bool | None = None,
    name: str | None = None,
    check: int = 6,
    taste: bool = False,
) -> dict[str, Any]:
    """Match, write ``refmatch.json``, optionally fit and save a learned preset.

    With ``taste`` the fit includes the personal taste head (needs the
    visual layer: ``cutpoint events`` on a ski project).
    """
    from cutpoint.config import remap_source  # noqa: PLC0415
    from cutpoint.events.common import load_highlights  # noqa: PLC0415

    hl = load_highlights(project)
    seen: dict[str, Path] = {}
    for s in hl["sources"]:
        seen.setdefault(s["path"], Path(s["path"]))
    sources = list(seen.values())
    ref = remap_source(reference)
    result = match_reference(ref, sources, params or MatchParams())
    result["reference"] = str(reference)
    _log_match(result)
    checks = check_frames({**result, "reference": str(ref)}, project / "refmatch_check", check)
    result["check_frames"] = [str(c) for c in checks]
    cache.save_json(project / "refmatch.json", result)
    logger.info("Wrote %s", project / "refmatch.json")
    if fit is None:
        fit = hl.get("style_kind") == "ski"
    if fit:
        from cutpoint.styles import get_style  # noqa: PLC0415

        base = get_style(hl["style"]).base
        rep = fit_weights(project, result, base=base, name=name, taste=taste)
        result["fit"] = rep
        cache.save_json(project / "refmatch.json", result)
        _log_fit(rep, get_style(base).weights)
    return result


def run_fit(
    projects: list[Path],
    *,
    name: str | None = None,
    base: str = "ski",
    taste: bool = True,
    do_ablate: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """``cutpoint fit``: fit one preset on several projects' ``refmatch.json`` at once.

    Reports in-sample, cross-validated (clips held out across all projects) and
    cross-project numbers (train on all but one project, test on that one),
    optionally with feature ablations, and writes ``<first project>/fit_report.json``.
    """
    datasets = [load_fit_data(p) for p in projects]
    for ds in datasets:
        logger.info(
            "%s: %d saved sources, %.0f s of positives, edit %.0f s",
            ds.name,
            len(ds.feats),
            sum(sp["end"] - sp["start"] for sp in ds.result["positives"]),
            ds.result["reference_duration"],
        )
    out: dict[str, Any] = {"projects": [str(p) for p in projects]}
    if do_ablate:
        out["ablation"] = ablate(datasets, base=base)
    rep = fit_weights(datasets, base=base, name=name, taste=taste, save=save)
    out["fit"] = rep
    from cutpoint.styles import get_style  # noqa: PLC0415

    if save:
        _log_fit(rep, get_style(base).weights)
    path = projects[0] / "fit_report.json"
    cache.save_json(path, out)
    logger.info("Wrote %s", path)
    return out


def _log_match(res: dict[str, Any]) -> None:
    logger.info(
        "Refmatch: %d of %d shots matched (%d pieces), coverage %.1f%% of %.1f s; "
        "%d positive spans (%.1f s) in %d of %d sources",
        res["n_matched"],
        res["n_shots"],
        res["n_pieces"],
        100 * res["coverage"],
        res["reference_duration"],
        len(res["positives"]),
        sum(p["end"] - p["start"] for p in res["positives"]),
        len(res["used_sources"]),
        len(res["used_sources"]) + len(res["unused_sources"]),
    )
    unmatched = [s for s in res["shots"] if not s["matched"]]
    if unmatched:
        logger.info(
            "Unmatched shots: %s",
            ", ".join(
                f"{s['ref_start']:.1f}-{s['ref_end']:.1f}s ({s.get('reason')})"
                for s in unmatched[:12]
            )
            + (" ..." if len(unmatched) > 12 else ""),
        )


def _log_fit(rep: dict[str, Any], default: dict[str, float]) -> None:
    learned = json.loads(Path(rep["preset"]).read_text(encoding="utf-8"))["weights"]
    logger.info(
        "Learned weights (default -> learned; per-feature AUC / Spearman on unmasked 1 s bins):"
    )
    for k in rep.get("features_fitted", default):
        st = rep["features"].get(k, {})
        logger.info(
            "  %-10s %.2f -> %.2f   AUC %s  rho %s",
            k,
            default.get(k, 0),
            learned.get(k, 0),
            st.get("auc"),
            st.get("spearman"),
        )
    a = rep["at_edit_duration"]
    cv = rep["cv_at_edit_duration"]
    logger.info(
        "At the edit's duration: default P %.2f R %.2f moments %d/%d | learned "
        "(in-sample) P %.2f R %.2f moments %d/%d",
        a["default"]["precision"],
        a["default"]["recall"],
        a["default"]["moments_hit"],
        a["default"]["moments"],
        a["learned_in_sample"]["precision"],
        a["learned_in_sample"]["recall"],
        a["learned_in_sample"]["moments_hit"],
        a["learned_in_sample"]["moments"],
    )
    logger.info(
        "Cross-validated (held-out clips): default P %.2f R %.2f | learned P %.2f "
        "R %.2f; AUC default %s learned-cv %s",
        cv["default"]["precision"],
        cv["default"]["recall"],
        cv["learned"]["precision"],
        cv["learned"]["recall"],
        rep["auc"]["default"],
        rep["auc"]["learned_cv"],
    )
    for c in rep.get("cross_project", []):
        logger.info(
            "Cross-project %s -> %s: default P %.2f R %.2f | learned P %.2f R %.2f "
            "(mask_friends %g)",
            c["train"],
            c["test"],
            c["default"]["precision"],
            c["default"]["recall"],
            c["learned"]["precision"],
            c["learned"]["recall"],
            c["mask_friends"],
        )
    m = rep["mask"]
    logger.info(
        "Positives: %.0f s; %.0f s inside lift/stop masks (%.0f%% of footage is "
        "masked), %.0f s eligible at mask_friends %g (0 = soft mask), %.0f s "
        "unreachable",
        m["positives_s"],
        m["in_lift_or_stop_s"],
        100 * m["share_of_footage_masked"],
        m["unlocked_by_vlm_friends_s"],
        rep["mask_friends"],
        m["still_masked_s"],
    )
    logger.info("Saved learned preset %s", rep["preset"])
