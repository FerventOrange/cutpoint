"""Layer 1d (voice): wind-robust speech and laughter grids from the cached 16 kHz audio.

Helmet- or chest-mounted action-cam audio is dominated by wind while riding.
PANNs on the raw track hears it as Explosion/Gunfire, and the voice classes
drown. This layer computes, on the 0.1 s analysis grid of each source:

* ``wind_db``: energy below 200 Hz relative to the voice band (300-3400 Hz),
  in dB, from a 32 ms STFT. Wind is broadband rumble with most of its energy
  at the bottom; voices are not.
* ``level_db``: voice-band level (dBFS), for diagnostics.
* ``vad``: Silero VAD v5 (ONNX on the CPU, MIT licence) speech probability on
  200 Hz high-passed audio, max per bin.
* PANNs Cnn14 framewise probabilities, max per bin, for the raw track
  (``raw_*``) and the 200 Hz high-passed one (``hp_*``): ``speech``,
  ``laugh``, ``cheer``, ``wind`` and ``boom`` (explosion/gunfire).

Raw grids are cached per source as ``cache/<source>/voice-<hash>.npz``;
:func:`voice_features` turns them into 0-1 ``laughter`` (threshold rising
with the wind level) and ``voice`` features, and silences the
explosion/gunfire classes while the camera moves.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.events import paths
from cutpoint.events.common import WAV_SR, Source, iter_audio_blocks, params_key

logger = logging.getLogger(__name__)

VOICE_VERSION = 1
DT = 0.1
GROUPS: dict[str, tuple[str, ...]] = {
    "speech": (
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Conversation",
        "Narration, monologue",
        "Babbling",
    ),
    "laugh": ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"),
    "cheer": ("Cheering", "Whoop", "Shout", "Yell", "Screaming", "Children shouting"),
    "wind": ("Wind", "Wind noise (microphone)"),
    "boom": ("Explosion", "Gunshot, gunfire", "Boom"),
}


@dataclass(frozen=True)
class VoiceParams:
    """Settings of the voice layer (all part of the cache key)."""

    highpass_hz: float = 200.0
    wind_lo_hz: float = 200.0  # wind band: below this
    voice_band: tuple[float, float] = (300.0, 3400.0)
    block_s: float = 60.0
    overlap_s: float = 2.0

    def key(self) -> dict[str, Any]:
        """Cache-key dict."""
        return {"layer": "voice", "v": VOICE_VERSION, **asdict(self)}


def cache_path(src: Source | Path, params: VoiceParams) -> Path:
    """``cache/<source>/voice-<hash>.npz``."""
    cdir = src.cdir if isinstance(src, Source) else cache.source_dir(src)
    return cdir / f"voice-{params_key(params.key())}.npz"


def load_voice(
    src: Source | Path, params: VoiceParams | None = None
) -> dict[str, np.ndarray] | None:
    """Cached voice grids of a source (absolute 0.1 s grid), or None if the layer has
    not run or the footage is offline (its cache key needs the file's size and mtime)."""
    try:
        path = cache_path(src, params or VoiceParams())
    except OSError as err:
        logger.warning("%s: footage not reachable (%s); no voice data for it", src, err)
        return None
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def _highpass(x: np.ndarray, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt  # noqa: PLC0415

    sos = butter(4, hz, "highpass", fs=WAV_SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _band_grids(x: np.ndarray, p: VoiceParams) -> tuple[np.ndarray, np.ndarray]:
    """``(wind_db, level_db)`` per 0.1 s bin of one block (32 ms Hann STFT, 10 ms hop)."""
    win, hop = 512, 160
    n = 1 + max(0, (len(x) - win) // hop)
    if len(x) < win:
        return np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n] * np.hanning(win)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    f = np.fft.rfftfreq(win, 1 / WAV_SR)
    lo = spec[:, (f >= 20) & (f < p.wind_lo_hz)].sum(1)
    mid = spec[:, (f >= p.voice_band[0]) & (f < p.voice_band[1])].sum(1)
    per = int(DT * WAV_SR / hop)  # 10 STFT frames per bin
    nb = n // per
    lo = lo[: nb * per].reshape(nb, per).mean(1)
    mid = mid[: nb * per].reshape(nb, per).mean(1)
    wind_db = 10 * np.log10((lo + 1e-9) / (mid + 1e-9))
    level_db = 10 * np.log10(mid / win**2 + 1e-12)
    return wind_db, level_db


class SileroVad:
    """Silero VAD v5 ONNX model, run on many independent 16 kHz streams at once."""

    CHUNK, CONTEXT = 512, 64

    def __init__(self) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        if not paths.silero_vad().exists():
            raise FileNotFoundError(
                f"{paths.silero_vad()} missing; run cutpoint setup --only voice"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(paths.silero_vad()), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def probs(self, x: np.ndarray, seg_s: float = 30.0, pre_s: float = 1.0) -> np.ndarray:
        """Speech probability per 32 ms chunk of ``x``.

        The audio is cut into ``seg_s`` segments, each preceded by ``pre_s`` of
        context, and the segments are run as one batch (the recurrent state is
        per row). Context chunks are dropped from the output.
        """
        c = self.CHUNK
        seg, pre = int(seg_s * WAV_SR) // c * c, int(pre_s * WAV_SR) // c * c
        n_chunks = len(x) // c
        if n_chunks == 0:
            return np.zeros(0, np.float32)
        starts = list(range(0, n_chunks * c, seg))
        rows = []
        for s in starts:
            a = max(0, s - pre)
            r = x[a : s + seg]
            rows.append((s - a, np.pad(r, (0, pre + seg - len(r)))))
        batch = np.stack([r for _, r in rows]).astype(np.float32)
        state = np.zeros((2, len(rows), 128), np.float32)
        ctx = np.zeros((len(rows), self.CONTEXT), np.float32)
        sr = np.array(WAV_SR, np.int64)
        out = np.zeros((len(rows), batch.shape[1] // c), np.float32)
        for k in range(batch.shape[1] // c):
            chunk = batch[:, k * c : (k + 1) * c]
            inp = np.concatenate([ctx, chunk], 1)
            p, state = self.sess.run(None, {"input": inp, "state": state, "sr": sr})
            out[:, k] = p[:, 0]
            ctx = chunk[:, -self.CONTEXT :]
        res = np.concatenate(
            [o[skip // c : skip // c + seg // c] for (skip, _), o in zip(rows, out, strict=True)]
        )
        return res[:n_chunks]


def _panns_classes(groups: dict[str, tuple[str, ...]]) -> tuple[list[int], list[str]]:
    from cutpoint.events.audio_events import load_labels  # noqa: PLC0415

    labels = load_labels()
    idx, owner = [], []
    for g, names in groups.items():
        for name in names:
            idx.append(labels.index(name))
            owner.append(g)
    return idx, owner


def compute_voice(
    src: Source, det: Any, vad: SileroVad, params: VoiceParams, force: bool = False
) -> dict[str, np.ndarray] | None:
    """Compute (or load) the voice grids for one source.

    Args:
        src: Source with cached 16 kHz audio.
        det: Loaded :class:`cutpoint.events.audio_events.PannsDetector`.
        vad: Loaded Silero VAD.
        params: Layer settings.
        force: Ignore the cache.
    """
    from cutpoint.events.audio_events import FRAME_S, upsample_2x  # noqa: PLC0415

    path = cache_path(src, params)
    if not force and (hit := load_voice(src, params)) is not None:
        return hit
    if not src.audio_parts:
        logger.warning("%s: no cached 16 kHz audio; run `cutpoint analyze` first", src.name)
        return None
    t0 = time.perf_counter()
    total = max(p.offset + p.duration for p in src.audio_parts)
    n = int(total / DT) + 1
    idx, owner = _panns_classes(GROUPS)
    owner_arr = np.array(owner)
    grids = {
        "wind_db": np.full(n, np.nan, np.float32),
        "level_db": np.full(n, -120.0, np.float32),
        "vad": np.zeros(n, np.float32),
    }
    for pre in ("raw", "hp"):
        for g in GROUPS:
            grids[f"{pre}_{g}"] = np.zeros(n, np.float32)
    edge_s = params.overlap_s / 2
    for t, block in iter_audio_blocks(src, params.block_s, params.overlap_s):
        hp = _highpass(block, params.highpass_hz)
        first = t <= 0
        last = t + len(block) / WAV_SR >= total - 0.05
        lo_s = 0.0 if first else edge_s  # block edges lack context
        hi_s = len(block) / WAV_SR - (0.0 if last else edge_s)

        # Only called within this iteration; the loop variables are bound explicitly.
        def put(
            name: str,
            values: np.ndarray,
            step: float,
            how: str = "max",
            *,
            t: float = t,
            lo_s: float = lo_s,
            hi_s: float = hi_s,
        ) -> None:
            times = t + np.arange(len(values)) * step
            ok = (times >= t + lo_s) & (times < t + hi_s)
            if how == "max":  # fine frames: floor into their bin
                rows = np.minimum((times[ok] / DT + 1e-6).astype(np.int64), n - 1)
            else:  # one value per bin: round (floor would skip bins on float error)
                rows = np.minimum(np.rint(times[ok] / DT).astype(np.int64), n - 1)
            if how == "max":
                np.maximum.at(grids[name], rows, values[ok])
            else:  # "set": one value per bin
                grids[name][rows] = values[ok]

        wind_db, level_db = _band_grids(block, params)  # raw: the wind band is < 200 Hz
        put("wind_db", wind_db, DT, "set")
        put("level_db", level_db, DT, "set")
        put("vad", vad.probs(hp), vad.CHUNK / WAV_SR)
        outs = det.framewise([upsample_2x(block), upsample_2x(hp)])
        for pre, fr in zip(("raw", "hp"), outs, strict=True):
            sel = fr[:, idx]
            for g in GROUPS:
                put(f"{pre}_{g}", sel[:, owner_arr == g].max(1), FRAME_S)
    grids["wind_db"] = np.nan_to_num(grids["wind_db"], nan=0.0)
    np.savez_compressed(path, **grids)
    logger.info(
        "%s: voice grids in %.1fs (%.0fx realtime)",
        src.name,
        time.perf_counter() - t0,
        total / max(1e-6, time.perf_counter() - t0),
    )
    return grids


def needs_run(sources: list[Source], params: VoiceParams, force: bool = False) -> bool:
    """Whether :func:`run_layer` would compute any grids (checks the cache files only)."""
    if force:
        return True
    for s in sources:
        try:
            if not cache_path(s, params).exists():
                return True
        except OSError:  # load_voice gives None: run_layer tries it
            return True
    return False


def run_layer(
    sources: list[Source], params: VoiceParams, force: bool = False, det: Any = None
) -> dict[str, Any]:
    """Compute the voice grids for every source that lacks them; returns stats.

    A given ``det`` (PANNs) stays open: the caller closes it."""
    need = [s for s in sources if force or load_voice(s, params) is None]
    stats: dict[str, Any] = {"cached": len(sources) - len(need), "computed": 0}
    if not need:
        return stats
    from cutpoint.events.audio_events import PannsDetector  # noqa: PLC0415

    own = det is None
    det = det or PannsDetector()
    stats["device"] = det.device
    vad = SileroVad()
    try:
        for s in need:
            if compute_voice(s, det, vad, params, force=True) is not None:
                stats["computed"] += 1
    finally:
        if own:
            det.close()
    return stats


# --------------------------------------------------------------------------- features
# Calibrated on helmet-mounted action-cam footage with wind noise (in-camera wind
# processing): speech probabilities do not false-fire in wind, they collapse
# (PANNs speech 0.43 when stopped, 0.02 above 35 km/h; whisper finds words in
# 1% of the seconds above 20 km/h), so the voice feature uses fixed
# thresholds. Laughter is quiet on this audio (PANNs laughter p99 0.001), so it
# is read on a log scale, with a threshold that rises with wind. The audio's
# own low-band ratio barely tracks speed there (r = 0.09; the camera already
# filters wind), so telemetry speed is the better wind level when present.
@dataclass(frozen=True)
class VoiceCalibration:
    """Thresholds turning the grids into 0-1 features.

    ``wind`` (0-1) is ``(wind_db - wind_lo_db) / (wind_hi_db - wind_lo_db)``,
    maxed with the optional "moving" signal. Laughter is
    ``(log10 p - (laugh_log_thr + laugh_wind * wind)) / laugh_log_span`` on the
    high-passed PANNs laughter/cheer probability; voice is the mean of Silero
    VAD and high-passed PANNs speech.
    """

    wind_lo_db: float = 5.0
    wind_hi_db: float = 25.0
    laugh_log_thr: float = -3.7
    laugh_wind: float = 1.0
    laugh_log_span: float = 2.0
    cheer_weight: float = 0.7
    smooth_s: float = 1.0


def wind_level(grids: dict[str, np.ndarray], cal: VoiceCalibration | None = None) -> np.ndarray:
    """0-1 wind level from ``wind_db`` (1 s running max, so gusts cover their bin)."""
    c = cal or VoiceCalibration()
    w = np.clip((grids["wind_db"] - c.wind_lo_db) / (c.wind_hi_db - c.wind_lo_db), 0, 1)
    return _smooth_max(w, max(1, int(1.0 / DT)))


def _smooth_max(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x
    pad = np.pad(x, (k // 2, k - k // 2 - 1), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad, k).max(1)


def voice_features(
    grids: dict[str, np.ndarray],
    moving: np.ndarray | None = None,
    cal: VoiceCalibration | None = None,
) -> dict[str, np.ndarray]:
    """0-1 ``laughter``, ``voice``, ``wind`` and gated ``boom`` on the grid.

    Args:
        grids: :func:`load_voice` output.
        moving: Optional 0-1 "camera is moving fast" signal on the same grid
            (e.g. telemetry speed / 40 km/h). It replaces the audio wind
            estimate where larger, and explosion/gunfire are zeroed while it
            is above 0.3 (on action-cam audio those are wind).
        cal: Thresholds.
    """
    c = cal or VoiceCalibration()
    w = wind_level(grids, c)
    n = len(w)
    m = None
    if moving is not None:
        m = np.zeros(n)
        k = min(n, len(moving))
        m[:k] = np.clip(moving[:k], 0, 1)
        w = np.maximum(w, m)
    k = max(1, int(c.smooth_s / DT))
    social = np.maximum(grids["hp_laugh"], c.cheer_weight * grids["hp_cheer"])
    thr = c.laugh_log_thr + c.laugh_wind * w
    lf = np.clip((np.log10(social + 1e-6) - thr) / c.laugh_log_span, 0, 1)
    voice = 0.5 * (grids["vad"] + grids["hp_speech"])
    boom = grids["hp_boom"].copy()
    if m is not None:
        boom[m > 0.3] = 0.0
    return {"laughter": _smooth_max(lf, k), "voice": _smooth_max(voice, k), "wind": w, "boom": boom}
