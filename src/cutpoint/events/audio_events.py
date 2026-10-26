"""Layer 1a: sound-event detection with PANNs (Cnn14_DecisionLevelMax).

The model wants 32 kHz audio. By default the toolkit's cached 16 kHz analysis
WAV is upsampled 2x, so the H: drive is never touched. The cached audio has no
content above 8 kHz, which slightly weakens the gunshot/explosion classes but
barely affects voice classes (laughter, screams). The audio is processed in
blocks, so RAM use is flat for multi-hour files.

Detection on single-track gameplay mixes: laughter under gunfire scores low in
absolute terms, so each class fires when its pooled probability exceeds
EITHER an absolute threshold OR a relative one (``rel_factor`` times the local
rolling median over ``baseline_s`` plus ``rel_margin``) with a lower floor.
"""

from __future__ import annotations

import csv
import importlib.util
import logging
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.events import paths
from cutpoint.events.common import Source, free_torch, iter_audio_blocks, layer_cache_path

logger = logging.getLogger(__name__)

MODEL_SR = 32000
FRAME_S = 0.01  # framewise output resolution of Cnn14_DecisionLevelMax (hop 320 @ 32 kHz)


@dataclass(frozen=True)
class ClassGroup:
    """AudioSet labels reported together under one name, with thresholds.

    Attributes:
        name: Event label written to events.json (e.g. "Laughter").
        labels: Exact AudioSet display names (class_labels_indices.csv).
        abs_thr: Fire when the pooled probability is at least this.
        rel_factor: ...or at least ``rel_factor * local_median + rel_margin``...
        rel_margin: ...(see above)...
        rel_floor: ...and at least this (keeps the relative rule out of noise).
        min_dur: Drop spans shorter than this (s).
        seed: Whether spans of this group seed VLM windows.
    """

    name: str
    labels: tuple[str, ...]
    abs_thr: float
    rel_factor: float = 3.0
    rel_margin: float = 0.04
    rel_floor: float = 0.08
    min_dur: float = 0.3
    seed: bool = True


# Calibrated on a 1080p tactical-shooter stream (game + voice chat on one track):
# genuine laughter peaks at only 0.04-0.13 there (0.4 in loud bursts) and the
# local median is ~0, so the relative rule effectively becomes ``rel_floor``.
DEFAULT_GROUPS: tuple[ClassGroup, ...] = (
    ClassGroup(
        "Laughter",
        ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"),
        abs_thr=0.12,
        rel_factor=4.0,
        rel_margin=0.03,
        rel_floor=0.04,
    ),
    ClassGroup(
        "Scream",
        ("Screaming", "Shout", "Yell", "Bellow", "Battle cry", "Children shouting"),
        abs_thr=0.15,
        rel_factor=4.0,
        rel_margin=0.02,
        rel_floor=0.03,
    ),
    ClassGroup(
        "Cheering",
        ("Cheering", "Whoop", "Applause"),
        abs_thr=0.15,
        rel_factor=4.0,
        rel_margin=0.015,
        rel_floor=0.02,
    ),
    ClassGroup(
        "Gunfire",
        ("Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun"),
        abs_thr=0.40,
        rel_factor=99.0,
        seed=False,
    ),
    ClassGroup("Explosion", ("Explosion", "Boom"), abs_thr=0.30, rel_factor=99.0, seed=False),
)


@dataclass
class AudioParams:
    """Tunable parameters for :func:`detect_audio_events` (all go into the cache key)."""

    pool_s: float = 0.32  # time resolution of the pooled probabilities
    block_s: float = 60.0  # seconds of audio per inference block
    overlap_s: float = 2.0  # context shared between blocks (edges discarded)
    batch: int = 4  # blocks per forward pass
    baseline_s: float = 60.0  # window of the rolling median baseline
    merge_gap_s: float = 0.7  # join spans separated by less than this
    groups: tuple[ClassGroup, ...] = field(default_factory=lambda: DEFAULT_GROUPS)

    def key(self) -> dict[str, Any]:
        """Parameters as a JSON-able dict for cache keys."""
        return {**asdict(self), "groups": [asdict(g) for g in self.groups]}


def load_labels() -> list[str]:
    """AudioSet display names in model output order."""
    with paths.panns_labels().open(encoding="utf-8") as fh:
        return [row["display_name"] for row in csv.DictReader(fh)]


def _import_panns_models() -> types.ModuleType:
    """Import ``panns_inference.models`` without running the package ``__init__``.

    The package ``__init__`` reads (and tries to download) the label CSV in
    ``~/panns_data``; we keep everything under ``models/panns`` instead.
    """
    if "panns_inference.models" in sys.modules:
        return sys.modules["panns_inference.models"]
    spec = importlib.util.find_spec("panns_inference")
    if spec is None or not spec.submodule_search_locations:
        from cutpoint.extras import missing_module_message  # noqa: PLC0415

        raise ImportError(missing_module_message("panns_inference"))
    if "panns_inference" not in sys.modules:
        pkg = types.ModuleType("panns_inference")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["panns_inference"] = pkg
    import panns_inference.models as models  # noqa: PLC0415

    return models


class PannsDetector:
    """Loaded Cnn14_DecisionLevelMax returning framewise class probabilities."""

    def __init__(self, device: str | None = None) -> None:
        import torch  # noqa: PLC0415

        if not paths.panns_ckpt().exists():
            raise FileNotFoundError(
                f"{paths.panns_ckpt()} missing; run cutpoint setup --only panns"
            )
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        models = _import_panns_models()
        self.model = models.Cnn14_DecisionLevelMax(
            sample_rate=MODEL_SR,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        ckpt = torch.load(paths.panns_ckpt(), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        if self.device == "cpu":
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        logger.info("PANNs Cnn14_DecisionLevelMax on %s", self.device)

    def framewise(self, batch: list[np.ndarray]) -> list[np.ndarray]:
        """Framewise probabilities ``(frames, 527)`` for each 32 kHz waveform."""
        torch = self.torch
        n = max(len(x) for x in batch)
        arr = np.zeros((len(batch), n), dtype=np.float32)
        for i, x in enumerate(batch):
            arr[i, : len(x)] = x
        with torch.inference_mode():
            out = self.model(torch.from_numpy(arr).to(self.device))["framewise_output"]
        probs = out.float().cpu().numpy()
        return [probs[i, : int(len(x) / MODEL_SR / FRAME_S)] for i, x in enumerate(batch)]

    def close(self) -> None:
        """Drop the model and release GPU memory."""
        del self.model
        free_torch()


def upsample_2x(x: np.ndarray) -> np.ndarray:
    """16 kHz -> 32 kHz polyphase resampling (scipy if present, else linear)."""
    try:
        from scipy.signal import resample_poly  # noqa: PLC0415

        return resample_poly(x, 2, 1).astype(np.float32)
    except ImportError:
        t = np.arange(len(x) * 2) / 2.0
        return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """Rolling median with a coarse stride (fast enough for hours of data)."""
    if len(x) == 0:
        return x
    stride = max(1, win // 8)
    centers = np.arange(0, len(x), stride)
    meds = np.array([np.median(x[max(0, c - win // 2) : c + win // 2 + 1]) for c in centers])
    return np.interp(np.arange(len(x)), centers, meds)


def _spans(
    mask: np.ndarray, probs: np.ndarray, dt: float, t0: float, gap: float, min_dur: float
) -> list[dict[str, float]]:
    """Merge a boolean mask into spans with peak/mean probability."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    spans: list[list[int]] = [[idx[0], idx[0]]]
    max_gap = max(1, round(gap / dt))
    for i in idx[1:]:
        if i - spans[-1][1] <= max_gap:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    out = []
    for a, b in spans:
        start, end = t0 + a * dt, t0 + (b + 1) * dt
        if end - start < min_dur:
            continue
        seg = probs[a : b + 1]
        peak = int(np.argmax(seg))
        out.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "peak": round(float(seg[peak]), 3),
                "peak_t": round(t0 + (a + peak) * dt, 2),
                "mean": round(float(seg.mean()), 3),
            }
        )
    return out


def pooled_probabilities(
    src: Source, det: PannsDetector, params: AudioParams, class_idx: list[int]
) -> tuple[np.ndarray, float]:
    """Run PANNs over the whole source; return ``(T, len(class_idx))`` pooled probs.

    Every block's frames are max-pooled into one absolute time grid, so
    overlapping blocks join seamlessly (block edges, which lack context, are
    dropped except at the very start and end of the audio).

    Returns:
        (pooled probabilities, time of row 0). Rows are ``params.pool_s`` apart.
    """
    dt = max(1, round(params.pool_s / FRAME_S)) * FRAME_S
    total = max((p.offset + p.duration for p in src.audio_parts), default=0.0)
    grid = np.zeros((int(total / dt) + 2, len(class_idx)), dtype=np.float32)
    edge = int(params.overlap_s / 2 / FRAME_S)
    pending: list[tuple[float, np.ndarray]] = []

    def flush() -> None:
        outs = det.framewise([upsample_2x(x) for _, x in pending])
        for (t, x), fr in zip(pending, outs, strict=True):
            lo = edge if t > 0 else 0
            is_last = t + len(x) / 16000 >= total - 0.05
            hi = len(fr) if is_last else max(lo, len(fr) - edge)
            if hi <= lo:
                continue
            times = t + np.arange(lo, hi) * FRAME_S
            rows = np.minimum((times / dt).astype(np.int64), len(grid) - 1)
            np.maximum.at(grid, rows, fr[lo:hi][:, class_idx])
        pending.clear()

    for t, block in iter_audio_blocks(src, params.block_s, params.overlap_s):
        pending.append((t, block))
        if len(pending) >= params.batch:
            flush()
    if pending:
        flush()
    return grid, 0.0


def cached_audio_events(src: Source, params: AudioParams) -> list[dict[str, Any]] | None:
    """Cached audio events of one source, or None on a cache miss (a hit may be empty)."""
    hit = cache.load_json(layer_cache_path(src.cdir, "audio", params.key()))
    return None if hit is None else hit["events"]


def detect_audio_events(
    src: Source, det: PannsDetector | None, params: AudioParams, force: bool = False
) -> list[dict[str, Any]]:
    """Detect audio events in one source (cached per source + parameters).

    Args:
        src: Source with cached 16 kHz audio.
        det: Loaded detector, or None to only read the cache (returns [] on miss).
        params: Detection parameters.
        force: Ignore the cache.

    Returns:
        Events ``{t, end, type: "audio", label, confidence, top_class, ...}``.
    """
    cpath = layer_cache_path(src.cdir, "audio", params.key())
    if not force and (hit := cached_audio_events(src, params)) is not None:
        return hit
    if det is None:
        return []
    if not src.audio_parts:
        logger.warning("%s: no cached 16 kHz audio; run `cutpoint analyze` first", src.name)
        return []
    labels = load_labels()
    class_idx: list[int] = []
    owner: list[int] = []  # group index of each selected class
    for gi, g in enumerate(params.groups):
        for lab in g.labels:
            if lab not in labels:
                raise ValueError(f"AudioSet label {lab!r} not in {paths.panns_labels()}")
            class_idx.append(labels.index(lab))
            owner.append(gi)
    t_start = time.perf_counter()
    probs, t0 = pooled_probabilities(src, det, params, class_idx)
    dt = max(1, round(params.pool_s / FRAME_S)) * FRAME_S
    win = max(3, int(params.baseline_s / dt))
    events: list[dict[str, Any]] = []
    owner_arr = np.array(owner)
    for gi, g in enumerate(params.groups):
        cols = np.flatnonzero(owner_arr == gi)
        gp = probs[:, cols].max(axis=1)
        top = probs[:, cols].argmax(axis=1)
        base = _rolling_median(gp, win)
        mask = (gp >= g.abs_thr) | (
            (gp >= g.rel_floor) & (gp >= g.rel_factor * base + g.rel_margin)
        )
        for sp in _spans(mask, gp, dt, t0, params.merge_gap_s, g.min_dur):
            i_peak = round((sp["peak_t"] - t0) / dt)
            i_peak = min(max(i_peak, 0), len(top) - 1)
            events.append(
                {
                    "t": sp["start"],
                    "end": sp["end"],
                    "type": "audio",
                    "label": g.name,
                    "confidence": sp["peak"],
                    "mean": sp["mean"],
                    "peak_t": sp["peak_t"],
                    "top_class": labels[class_idx[cols[top[i_peak]]]],
                    "baseline": round(float(base[i_peak]), 3),
                    "relative": bool(sp["peak"] < g.abs_thr),
                    "seed": g.seed,
                }
            )
    events.sort(key=lambda e: e["t"])
    elapsed = time.perf_counter() - t_start
    logger.info(
        "%s: %d audio events in %.1fs (%.0fx realtime)",
        src.name,
        len(events),
        elapsed,
        src.duration / max(elapsed, 1e-6),
    )
    cache.save_json(
        cpath,
        {
            "source": str(src.path),
            "params": params.key(),
            "elapsed_s": round(elapsed, 2),
            "events": events,
        },
    )
    return events
