"""Crash / wipeout detection for action-cam footage (``cutpoint crashes``).

The older telemetry rule (:mod:`cutpoint.telemetry`: one impact >= 4 g
followed by a GPS stop) misses the crashes that matter most: a long tumble
keeps sliding down the slope at speed, so there is no stop right after the
biggest impact, and the accelerometer clips at 8 g per axis. This module
combines four independent kinds of evidence per candidate window:

* **telemetry** (full-rate GPMF IMU, cached per source as
  ``cache/<source>/crashimu-v<N>.npz`` on the 0.1 s grid; files on network storage are read
  through a GPMF-only remux of a few MB): repeated hard
  impacts and axis saturation, extreme vibration, off-axis rotation (pitch /
  roll; looking around is yaw), the camera's gravity vector turning away from
  its riding orientation (upside down, sideways), a speed collapse and lying
  still afterwards;
* **visual**: CLIP ViT-L/14 zero-shot crash prompts ("tumbling view of sky and
  snow", "camera buried in snow", "skier lying in the snow after a fall", ...)
  against normal riding, on the cached ~1 fps keyframe embeddings of the visual
  layer (``cutpoint events``);
* **audio**: PANNs yell / scream and loudness from the cached voice layer, and
  laughter in the aftermath;
* **transcript**: crash words in the next few minutes ("you good?", "tumble",
  "yard sale", "ate it", "fell", ...).

A 1 fps keyframe sample is too coarse for a 1-2 s tumble, so the best
candidates are **confirmed densely**: their window is decoded at ``dense_fps``
(NVDEC seek, about 10 s of video each) and every frame is embedded with CLIP;
the dense crash-prompt score and the frame-to-frame view change (a tumble
spins the view; riding changes it smoothly) are cached per source and
window (``cache/<source>/crashdense-<hash>.json``).

Results go to ``<project>/crashes.json``: every candidate with its signals,
the five sub-scores (telemetry, visual, dense, audio, words), ``confidence``
(0-1), and the clip it deserves (lead-in
from ``lead_s`` before the first impact, through the aftermath: until the rider
is moving again, the talk about it ends or ``max_after_s``). ``cutpoint analyze
--style ski`` reads this file: candidates at or above the style's
``must_include_crash`` confidence become must-include clips that ``cutpoint clips``
always exports (rare events are underweighted by learned taste weights), and
the ``wipeout`` feature carries the confidence on the grid.

Nothing here is ski-specific except the visual prompts and the transcript
words, which come from the project's activity profile (:mod:`cutpoint.activity`:
``--activity``, ``--context`` or detected from the footage); impacts, rotation,
gravity and stops work for any GoPro HERO 9-13 sport. Prompts passed through
:class:`CrashParams` override the profile.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import activity, cache
from cutpoint.dsp import conv_same
from cutpoint.stage import is_remote as _is_remote

logger = logging.getLogger(__name__)

DT = 0.1
G = 9.80665
CRASH_VERSION = 2  # 2: CLIP built with QuickGELU (dense scores and crash keys changed)
IMU_VERSION = 1
CRASHES_FILE = "crashes.json"
CHECK_DIR = "crash_check"
# On-lift damping reaches this far past the event: the lift line and boarding
# jostle the camera just before the ride, and the GPS lift mask starts up to
# ~1 s after the chair departs.
LIFT_LEAD_S = 5.0

# Zero-shot prompts (CLIP ViT-L/14, the visual layer's model) and transcript
# cues (phrase -> weight, prefix match with a trailing ``*``) of the built-in
# ski activity profile (presets/activities/ski.json on the shared base). Other
# footage gets its own profile (:mod:`cutpoint.activity`).
_SKI = activity.full(activity.builtin("ski"))["incident"]
CRASH_POS: tuple[str, ...] = tuple(_SKI["pos"])
CRASH_NEG: tuple[str, ...] = tuple(_SKI["neg"])
CRASH_WORDS: dict[str, float] = dict(_SKI["words"])


# --------------------------------------------------------------------------- params
@dataclass(frozen=True)
class CrashParams:
    """Thresholds and weights (the detection part is in the cache key).

    Attributes:
        impact_g: A 0.1 s bin at or above this |a| is an impact.
        hard_g: A "hard" impact (counted for tumbles).
        cluster_gap_s: Shocks closer than this belong to one event (events
            closer than 3 s are merged too: a tumble keeps going).
        still_kmh: GPS speed that counts as stopped.
        lead_s: Clip lead-in before the first impact (the approach).
        min_after_s: Aftermath always kept after the last impact.
        max_after_s: Aftermath cap after the last impact.
        max_clip_s: Total clip length cap.
        dense_fps: Frame rate of the confirm decode.
        dense_pre_s: Confirm window before the first impact.
        dense_post_s: Confirm window after the last impact.
        dense_top: Candidates per project that get the dense confirm.
        prefilter: Candidates below this pre-score (telemetry + visual + audio
            + transcript, before the dense step) are not confirmed.
        weights: Fusion weights of the sub-scores.
        pos, neg: Zero-shot prompts.
        words: Transcript cues.
    """

    impact_g: float = 4.0
    hard_g: float = 6.0
    cluster_gap_s: float = 1.5
    still_kmh: float = 3.0
    lead_s: float = 5.0
    min_after_s: float = 8.0
    max_after_s: float = 30.0
    max_clip_s: float = 40.0
    dense_fps: float = 5.0
    dense_pre_s: float = 4.0
    dense_post_s: float = 6.0
    dense_top: int = 15
    prefilter: float = 0.2
    weights: tuple[tuple[str, float], ...] = (
        ("telemetry", 0.40),
        ("visual", 0.15),
        ("dense", 0.20),
        ("audio", 0.10),
        ("words", 0.15),
    )
    pos: tuple[str, ...] = CRASH_POS
    neg: tuple[str, ...] = CRASH_NEG
    words: tuple[tuple[str, float], ...] = tuple(CRASH_WORDS.items())

    def key(self) -> str:
        """Short hash of everything that changes results (+ code version)."""
        return cache.params_hash({**asdict(self), "v": CRASH_VERSION})

    def dense_key(self) -> str:
        """Hash of what the dense decode depends on (prompts, fps)."""
        return cache.params_hash(
            {"fps": self.dense_fps, "pos": self.pos, "neg": self.neg, "v": CRASH_VERSION}
        )

    def with_profile(self, profile: dict[str, Any]) -> CrashParams:
        """These params with an activity profile's incident prompts and words."""
        inc = profile["incident"]
        return replace(
            self, pos=tuple(inc["pos"]), neg=tuple(inc["neg"]), words=tuple(inc["words"].items())
        )


# --------------------------------------------------------------------------- IMU grids
def _gpmf_copy(src: Path, tmp: Path) -> Path | None:
    """GPMF-only MP4 of ``src`` in ``tmp`` (ffmpeg reads just that track's chunks).

    telemetry-parser reads most of the file it is given; remuxing the "GoPro
    MET" track alone reads a few MB instead of GBs from slow or network storage.
    """
    from cutpoint.ffmpeg_utils import FFmpegError, ffprobe_json, run  # noqa: PLC0415

    try:
        info = ffprobe_json(src, entries="stream=index,codec_tag_string", timeout=60)
    except (OSError, FFmpegError):
        return None
    streams = info.get("streams") or []
    idx = next((s.get("index") for s in streams if s.get("codec_tag_string") == "gpmd"), None)
    if idx is None:
        return None
    dst = tmp / f"{src.stem}.gpmf.mp4"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        f"0:{idx}",
        "-c",
        "copy",
        "-copy_unknown",
        "-metadata:s:0",
        "handler_name=GoPro MET",
        "-f",
        "mov",
        str(dst),
    ]
    try:
        run(cmd, timeout=600)
    except (OSError, FFmpegError):
        return None
    return dst if dst.exists() and dst.stat().st_size > 0 else None


def _parse_imu(
    path: Path, force: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]] | None:
    """``(t_s, accel m/s^2 (N, 3), gyro deg/s (N, 3), telemetry groups)`` or None.

    ``force`` (``crashes --force``) logs a parse failure at INFO: the grids are
    then recomputed before ``telemetry.extract`` runs, which may hit its cache.
    """
    import telemetry_parser as tp_mod  # noqa: PLC0415

    try:
        parser = tp_mod.Parser(str(path))
        groups = parser.telemetry()
        imu = parser.normalized_imu()
    except Exception as err:  # noqa: BLE001 - the Rust parser raises plain Exceptions
        # Unforced: DEBUG, as telemetry.extract (run first on the same source) logged it
        # at INFO when it parsed the file.
        level = logging.INFO if force else logging.DEBUG
        logger.log(level, "%s: no readable GoPro telemetry (%s)", path.name, err)
        return None
    imu = [s for s in imu or [] if s.get("accl") is not None and s.get("gyro") is not None]
    if len(imu) < 10:
        return None
    t = np.array([s["timestamp_ms"] for s in imu]) / 1000.0
    acc = np.array([s["accl"] for s in imu], dtype=float)
    gyr = np.array([s["gyro"] for s in imu], dtype=float)
    return t, acc, gyr, groups


def _full_scale(groups: list[Any], key: str) -> float | None:
    """Per-axis full scale of a GPMF stream in its own unit (int16 max / SCAL)."""
    for g in groups:
        v = g.get(key)
        if isinstance(v, dict) and v.get("Scale"):
            sc = v["Scale"]
            sc = sc[0] if isinstance(sc, list | tuple) else sc
            try:
                return 32767.0 / float(sc)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
    return None


def _is_insta360(groups: list[Any]) -> bool:
    """True for telemetry-parser output of an Insta360 file (trailer metadata)."""
    return any(
        isinstance(g.get("Default"), dict) and "Metadata" in g["Default"] for g in groups[:2]
    )


def _grav_stream(groups: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Camera-fused gravity vector (GRAV, HERO 9+): times and unit vectors."""
    ts, vs = [], []
    pays = []
    for g in groups:
        v = g.get("GravityVector")
        if isinstance(v, dict) and "TimestampUs" in v and v.get("Data"):
            pays.append((int(v["TimestampUs"]) / 1e6, v["Data"]))
    for i, (t, data) in enumerate(pays):
        span = (pays[i + 1][0] - t) if i + 1 < len(pays) else len(data) / 30.0
        for k, d in enumerate(data):
            if isinstance(d, dict):
                ts.append(t + k * span / len(data))
                vs.append((d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)))
    if not ts:
        return np.zeros(0), np.zeros((0, 3))
    v = np.asarray(vs, float)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return np.asarray(ts), v


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:

    k = max(1, k)
    return np.stack([conv_same(x[:, i], np.ones(k) / k) for i in range(x.shape[1])], 1)


def _compute_imu(
    read_path: Path, duration: float, force: bool = False
) -> dict[str, np.ndarray] | None:
    parsed = _parse_imu(read_path, force)
    if parsed is None:
        return None
    t, acc, gyr, groups = parsed
    n = max(1, int(duration / DT) + 1)
    fs = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    idx = np.clip((t / DT).astype(int), 0, n - 1)
    am = np.linalg.norm(acc, axis=1) / G
    fs_acc = _full_scale(groups, "Accelerometer")
    if _is_insta360(groups):  # X3: the stream's full scale is in g (+-32 g per axis)
        sat = np.abs(acc).max(1) >= 0.985 * (fs_acc or 32.0) * G
    elif fs_acc:
        sat = np.abs(acc).max(1) >= 0.985 * fs_acc
    else:
        sat = am >= 7.9
    gdeg = np.linalg.norm(gyr, axis=1)  # normalized_imu gives deg/s
    # Off-axis rotation: the gyro component perpendicular to gravity (1 s
    # low-pass of the accelerometer). Looking around / turning is yaw about
    # gravity; a fall pitches and rolls the head.
    grav_lp = _lowpass(acc, int(fs))
    grav_lp /= np.linalg.norm(grav_lp, axis=1, keepdims=True) + 1e-9
    yaw = (gyr * grav_lp).sum(1)
    perp = np.linalg.norm(gyr - yaw[:, None] * grav_lp, axis=1)
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 0.05)

    def bmax(v: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        np.maximum.at(out, idx, v)
        return out

    def bsum(v: np.ndarray) -> np.ndarray:
        return np.bincount(idx, weights=v, minlength=n)[:n]

    grids = {
        "acc_g": bmax(am),
        "sat": bmax(sat.astype(float)),
        "gyro_dps": bmax(gdeg),
        "rot_deg": bsum(gdeg * dt),
        "offaxis_deg": bsum(perp * dt),
    }
    # Gravity direction: the camera's fused GRAV stream when present (robust
    # while tumbling), else a 0.5 s low-pass of the accelerometer.
    gt, gv = _grav_stream(groups)
    if len(gt) < 10:
        gt, gv = t, _lowpass(acc, int(fs / 2))
        gv /= np.linalg.norm(gv, axis=1, keepdims=True) + 1e-9
    gi = np.clip((gt / DT).astype(int), 0, n - 1)
    cnt = np.bincount(gi, minlength=n)[:n]
    for k, axis in enumerate("xyz"):
        s = np.bincount(gi, weights=gv[:, k], minlength=n)[:n]
        grids[f"grav_{axis}"] = np.divide(s, cnt, out=np.full(n, np.nan), where=cnt > 0)
    grids["imu_hz"] = np.array([fs])
    return grids


def imu_grids(
    src: Path, duration: float, force: bool = False, tmp_dir: Path | None = None
) -> dict[str, np.ndarray] | None:
    """Crash-relevant IMU grids of one source (0.1 s bins), cached per source.

    Grids: ``acc_g`` (peak |a|, g), ``sat`` (1 if an accelerometer axis hit
    full scale), ``gyro_dps`` (peak rotation rate), ``rot_deg`` / ``offaxis_deg``
    (rotation per bin: total, and perpendicular to gravity), ``grav_x/y/z``
    (camera gravity direction, unit vector; NaN without data).

    Args:
        src: Source (original path; ``source_remap`` and the stage apply).
        duration: Container duration (sets the grid length).
        force: Ignore the cache.
        tmp_dir: Where the GPMF-only remux of a remote file goes.
    """
    from cutpoint.config import remap_source  # noqa: PLC0415
    from cutpoint.stage import staged_path  # noqa: PLC0415

    path = cache.source_dir(src) / f"crashimu-v{IMU_VERSION}.npz"
    if not force and path.exists():
        with np.load(path) as z:
            if "none" in z.files:
                return None
            return {k: z[k].astype(np.float64) for k in z.files}
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    io = remap_source(src)
    x3_imu = imu_source(io)  # Insta360 X3: the _00_.insv trailer (read cheaply)
    read_path = x3_imu or Path(staged_path(str(io)))
    t0 = time.perf_counter()
    grids = None
    if x3_imu is None and read_path == io and _is_remote(io):
        with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
            small = _gpmf_copy(io, Path(tmp))
            if small is not None:
                grids = _compute_imu(small, duration, force)
    if grids is None:
        grids = _compute_imu(read_path, duration, force)
    if grids is None:
        np.savez_compressed(path, none=np.zeros(1))
        return None
    np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in grids.items()})
    logger.info(
        "%s: crash IMU grids in %.1fs (%.0f Hz)",
        src.name,
        time.perf_counter() - t0,
        float(grids["imu_hz"][0]),
    )
    return grids


# --------------------------------------------------------------------------- dense confirm
def decode_window(
    path: Path, t0: float, t1: float, fps: float, width: int = 448
) -> tuple[np.ndarray, np.ndarray]:
    """Frames of ``[t0, t1]`` at ``fps`` (one seek, NVDEC when available).

    Returns:
        ``(times (N,), frames (N, H, W, 3) uint8 RGB)``; empty arrays on failure.
    """
    from cutpoint.ffmpeg_utils import FFmpegError, decode_raw, probe  # noqa: PLC0415

    info = probe(path)
    height = round(width * info.height / max(1, info.width) / 2) * 2
    t0 = max(0.0, t0)
    dur = max(0.2, min(t1, info.duration) - t0)
    try:
        got = list(
            decode_raw(
                path,
                size=(width, height),
                vf=f"scale={width}:{height}:flags=area,fps={fps:g},format=rgb24",
                vf_cuda=f"scale_cuda={width}:{height},hwdownload,format=nv12,fps={fps:g},format=rgb24",
                start=t0,
                duration=dur,
                stamp_fps=fps,
                timeout=300,
            )
        )
    except FFmpegError as err:
        logger.debug("dense decode %s %.1f-%.1fs failed: %s", path.name, t0, t0 + dur, err)
        return np.zeros(0), np.zeros((0, height, width, 3), np.uint8)
    return np.array([t for t, _ in got]), np.stack([f for _, f in got])


class ClipEncoder:
    """CLIP ViT-L/14 image tower only (the visual layer's model and weights)."""

    def __init__(self, device: str | None = None) -> None:
        from cutpoint.events import paths  # noqa: PLC0415
        from cutpoint.events.visual import (  # noqa: PLC0415
            CLIP_ARCH,
            CLIP_MEAN,
            CLIP_PRETRAINED,
            CLIP_STD,
        )

        paths.apply_torch_home()
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_ARCH, pretrained=CLIP_PRETRAINED, cache_dir=str(paths.clip_dir())
        )
        self.model = model.eval().to(self.device, self.dtype)
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

    def embed(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        """``(N, 3, 768)`` L2-normalised embeddings of left / centre / right crops."""
        torch = self.torch
        out = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i : i + batch]).to(self.device).permute(0, 3, 1, 2)
            x = x.float() / 255
            b, _, h, w = x.shape
            s = min(h, w)
            offs = [0, (w - s) // 2, w - s]
            c = torch.cat([x[:, :, :, o : o + s] for o in offs], 0)
            c = torch.nn.functional.interpolate(
                c, size=(224, 224), mode="bicubic", antialias=True, align_corners=False
            )
            c = ((c.clamp(0, 1) - self.mean) / self.std).to(self.dtype)
            with torch.inference_mode():
                e = self.model.encode_image(c).float()
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.view(3, b, -1).permute(1, 0, 2).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3, 768), np.float32)

    def close(self) -> None:
        """Release the model and GPU memory."""
        from cutpoint.events.common import free_torch  # noqa: PLC0415

        del self.model
        free_torch()


def crash_prompt_scores(emb: np.ndarray, params: CrashParams) -> np.ndarray:
    """Zero-shot crash share per frame (mean over the crops), ``emb`` (N, 3, 768)."""
    from cutpoint.events.visual import prompt_score  # noqa: PLC0415

    if not len(emb):
        return np.zeros(0)
    return prompt_score(emb, params.pos, params.neg).mean(1)


def view_change(emb: np.ndarray) -> np.ndarray:
    """1 - cosine similarity of consecutive frames (centre crop), first frame 0."""
    if len(emb) < 2:
        return np.zeros(len(emb))
    c = emb[:, 1].astype(np.float32)
    sim = (c[1:] * c[:-1]).sum(1)
    return np.r_[0.0, 1.0 - sim]


# --------------------------------------------------------------------------- per-source inputs
@dataclass
class SourceInputs:
    """Everything cached about one source that the detector reads."""

    path: Path
    duration: float
    tel: Any = None  # cutpoint.telemetry.Telemetry
    imu: dict[str, np.ndarray] | None = None
    voice: dict[str, np.ndarray] | None = None
    visual: Any = None  # cutpoint.events.visual.VisualData
    words: list[dict[str, Any]] | None = None
    hits: list[tuple[float, float, str]] = field(default_factory=list)
    flip: np.ndarray | None = None
    kf_crash: np.ndarray | None = None
    lift: np.ndarray | None = None  # telemetry lift mask | 360 CLIP lift rides

    @property
    def name(self) -> str:
        """File name."""
        return self.path.name


def load_inputs(
    src: Path, duration: float, params: CrashParams, tmp_dir: Path | None = None
) -> SourceInputs:
    """Gather the cached layers of one source (computing only the IMU grids)."""
    from cutpoint.analyze import keyword_hits  # noqa: PLC0415
    from cutpoint.events import visual as V  # noqa: PLC0415
    from cutpoint.events import voice as VO  # noqa: PLC0415
    from cutpoint.mix import lift_rides  # noqa: PLC0415
    from cutpoint.sources import source_words  # noqa: PLC0415
    from cutpoint.telemetry import extract  # noqa: PLC0415

    inp = SourceInputs(path=src, duration=duration)
    inp.tel = extract(src, duration)
    inp.imu = imu_grids(src, duration, tmp_dir=tmp_dir)
    inp.voice = VO.load_voice(src)
    inp.visual = V.load_visual(src)
    inp.words = source_words(src)
    if inp.words:
        inp.hits = keyword_hits(inp.words, dict(params.words))
    if inp.imu is not None:
        inp.flip = _flip_deg(inp)
    if inp.visual is not None and len(inp.visual.t):
        inp.kf_crash = crash_prompt_scores(inp.visual.emb, params)
    tel_lift = _grid(inp, "lift_mask")
    n = len(tel_lift) if tel_lift is not None else max(1, int(duration / DT) + 1)
    rides = lift_rides(src, n)
    if rides is not None:
        inp.lift = rides if tel_lift is None else np.maximum(tel_lift, rides)
    return inp


def _flip_deg(inp: SourceInputs) -> np.ndarray:
    """Angle (deg) between the camera's gravity direction and its riding orientation.

    The riding orientation is the median gravity direction while moving
    (> 10 km/h, not on a lift); without GPS, the median over the whole clip.
    """
    g = inp.imu
    grav = np.stack([g["grav_x"], g["grav_y"], g["grav_z"]], 1)
    ok = np.isfinite(grav).all(1)
    sel = ok.copy()
    if inp.tel is not None:
        n = min(len(sel), inp.tel.n)
        sp = np.nan_to_num(inp.tel.grid["speed_kmh"][:n])
        moving = np.zeros(len(sel), bool)
        moving[:n] = (sp > 10) & (inp.tel.grid["lift_mask"][:n] == 0)
        if (ok & moving).sum() > 100:
            sel = ok & moving
    if not sel.any():
        return np.zeros(len(grav))
    base = np.median(grav[sel], 0)
    base /= np.linalg.norm(base) + 1e-9
    cos = np.nan_to_num(grav @ base, nan=1.0)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _grid(inp: SourceInputs, key: str) -> np.ndarray | None:
    if inp.tel is None or key not in inp.tel.grid:
        return None
    return inp.tel.grid[key]


def _seg(x: np.ndarray | None, a: float, b: float) -> np.ndarray:
    """Finite values of a 0.1 s grid between ``a`` and ``b`` seconds."""
    if x is None:
        return np.zeros(0)
    i, j = max(0, int(a / DT)), min(len(x), int(b / DT) + 1)
    s = np.asarray(x[i:j], float) if j > i else np.zeros(0)
    return s[np.isfinite(s)]


def _smax(x: np.ndarray | None, a: float, b: float, default: float = 0.0) -> float:
    s = _seg(x, a, b)
    return float(s.max()) if s.size else default


def _smed(x: np.ndarray | None, a: float, b: float) -> float | None:
    s = _seg(x, a, b)
    return float(np.median(s)) if s.size else None


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# --------------------------------------------------------------------------- telemetry
def _shock_spans(inp: SourceInputs, p: CrashParams) -> list[tuple[float, float]]:
    """``(first, last)`` shock times: impacts, very fast spins, off-axis rotation."""
    g = inp.imu
    off1 = np.convolve(g["offaxis_deg"], np.ones(10), mode="same")  # deg per 1 s
    shock = (g["acc_g"] >= p.impact_g) | (g["gyro_dps"] >= 900) | (off1 >= 150)
    idx = np.flatnonzero(shock)
    spans: list[list[int]] = []
    gap = int(p.cluster_gap_s / DT)
    for i in idx:
        if spans and i - spans[-1][1] <= gap:
            spans[-1][1] = int(i)
        else:
            spans.append([int(i), int(i)])
    return [(a * DT, b * DT) for a, b in spans]


def telemetry_features(inp: SourceInputs, t0: float, t1: float, p: CrashParams) -> dict[str, Any]:
    """Telemetry evidence for an event from ``t0`` (first shock) to ``t1`` (last)."""
    g = inp.imu or {}
    off1 = np.convolve(g["offaxis_deg"], np.ones(10), mode="same") if "offaxis_deg" in g else None
    acc = g.get("acc_g")
    speed, vib = _grid(inp, "speed_kmh"), _grid(inp, "vibration")
    still = _grid(inp, "still_mask")
    lift = inp.lift if inp.lift is not None else _grid(inp, "lift_mask")
    hard = int((_seg(acc, t0, t1) >= p.hard_g).sum())
    sat = int(_seg(g.get("sat"), t0, t1).sum())
    v_before = _smed(speed, t0 - 4.0, t0 - 0.5)
    v_peak_before = _smax(speed, t0 - 5.0, t0, default=float("nan"))
    sp_after = _seg(speed, t1, t1 + 10.0)
    v_min_after = float(sp_after.min()) if sp_after.size else None
    still_after = float(np.mean(_seg(still, t1, t1 + 15.0))) if still is not None else 0.0
    vib_before = _smed(vib, t0 - 4.0, t0 - 0.5) or 0.0
    f = {
        "peak_g": round(_smax(acc, t0, t1), 2),
        "hard_impacts": hard,
        "saturated": sat,
        "gyro_dps": round(_smax(g.get("gyro_dps"), t0, t1), 0),
        "offaxis_deg_1s": round(_smax(off1, t0 - 0.5, t1 + 0.5), 0),
        "vibration_g": round(_smax(vib, t0, t1), 2),
        "flip_deg": round(_smax(inp.flip, t0, t1 + 5.0), 0),
        "flip_after_deg": round(_smed(inp.flip, t1 + 1.0, t1 + 5.0) or 0.0, 0),
        "speed_before_kmh": None if v_before is None else round(v_before, 1),
        "speed_min_after_kmh": None if v_min_after is None else round(v_min_after, 1),
        "still_after": round(still_after, 2),
        "on_lift": bool(_smax(lift, t0, t1 + LIFT_LEAD_S) > 0),
    }
    # Moving before the event: GPS, else riding vibration (camera handling while
    # standing spins and flips the camera too).
    if v_before is not None:
        moving = _clip01((max(v_before, np.nan_to_num(v_peak_before)) - 4.0) / 8.0)
    else:
        moving = _clip01((vib_before - 0.06) / 0.1)
    stopped = 0.0
    if v_min_after is not None and v_before is not None:
        stopped = float(v_min_after <= p.still_kmh + 1.0)
    stopped = max(stopped, still_after)
    parts = {
        "impact": _clip01((f["peak_g"] - 5.0) / 5.0),
        "repeat": _clip01((hard - 1) / 5.0),
        "saturation": _clip01((sat - 1) / 3.0),
        "spin": max(
            _clip01((f["offaxis_deg_1s"] - 150.0) / 200.0), _clip01((f["gyro_dps"] - 900.0) / 900.0)
        ),
        "vibration": _clip01((f["vibration_g"] - 1.2) / 0.8),
        "flip": max(
            _clip01((f["flip_deg"] - 70.0) / 60.0),
            _clip01((f["flip_after_deg"] - 45.0) / 45.0) * stopped,
        ),
        "stop": stopped,
    }
    w = {
        "impact": 0.15,
        "repeat": 0.2,
        "saturation": 0.1,
        "spin": 0.2,
        "vibration": 0.15,
        "flip": 0.15,
        "stop": 0.15,
    }
    raw = sum(w[k] * v for k, v in parts.items())
    score = _clip01(raw) * (0.3 + 0.7 * moving) * (0.2 if f["on_lift"] else 1.0)
    f["parts"] = {k: round(v, 2) for k, v in parts.items()}
    f["moving_before"] = round(moving, 2)
    f["score"] = round(score, 3)
    return f


# --------------------------------------------------------------------------- other evidence
def visual_feature(inp: SourceInputs, t0: float, t1: float) -> dict[str, Any]:
    """Keyframe (~1 fps) zero-shot crash score around the event."""
    if inp.kf_crash is None:
        return {"score": 0.0, "kf_max": None}
    t = inp.visual.t.astype(float)
    sel = (t >= t0 - 0.5) & (t <= t1 + 5.0)
    if not sel.any():
        return {"score": 0.0, "kf_max": None}
    vals = np.sort(inp.kf_crash[sel])[::-1]
    top = float(vals[:2].mean())  # two frames, so one odd frame does not decide
    return {
        "score": round(_clip01((top - 0.6) / 0.3), 3),
        "kf_max": round(float(vals[0]), 3),
        "kf_top2": round(top, 3),
    }


def audio_feature(inp: SourceInputs, t0: float, t1: float) -> dict[str, Any]:
    """Yell / scream at the event and laughter in the aftermath (voice layer)."""
    v = inp.voice
    if v is None:
        return {"score": 0.0}
    yell = np.maximum(v["raw_cheer"], v["hp_cheer"])
    ref_y = max(float(np.percentile(yell, 99)), 1e-4)
    ref_l = max(float(np.percentile(v["hp_laugh"], 99)), 1e-4)
    y = _smax(yell, t0 - 1.0, t1 + 6.0)
    lg = _smax(v["hp_laugh"], t1, t1 + 30.0)
    s_y = _clip01(np.log10(max(y, 1e-6) / ref_y) / np.log10(4.0) + 0.5)
    s_l = _clip01(np.log10(max(lg, 1e-6) / ref_l) / np.log10(4.0) + 0.5)
    return {
        "score": round(max(s_y, 0.7 * s_l), 3),
        "yell": round(y, 4),
        "laugh_after": round(lg, 4),
    }


def words_feature(
    inp: SourceInputs, t0: float, t1: float, horizon_s: float = 240.0
) -> dict[str, Any]:
    """Crash words said during and after the event (decaying over ~90 s)."""
    total, found = 0.0, []
    for t, w, phrase in inp.hits:
        if t0 - 1.0 <= t <= t1 + horizon_s:
            total += w * float(np.exp(-max(0.0, t - t1) / 90.0))
            found.append((round(t, 1), phrase))
    return {
        "score": round(1.0 - float(np.exp(-total)), 3),
        "sum": round(total, 2),
        "phrases": found[:12],
    }


def _fuse(scores: dict[str, float], weights: dict[str, float], use: tuple[str, ...]) -> float:
    tot = sum(weights[k] for k in use)
    return sum(weights[k] * scores.get(k, 0.0) for k in use) / max(tot, 1e-9)


# --------------------------------------------------------------------------- candidates
def candidate_events(inp: SourceInputs, p: CrashParams) -> list[dict[str, Any]]:
    """Event spans of one source: telemetry shocks, plus strong spoken cues with
    no telemetry event before them (a friend's fall, a crash without GPMF)."""
    cands: list[dict[str, Any]] = []
    if inp.imu is not None:
        merged: list[list[float]] = []
        for a, b in _shock_spans(inp, p):
            if merged and a - merged[-1][1] <= 3.0:  # a tumble keeps going
                merged[-1][1] = b
            else:
                merged.append([a, b])
        cands += [{"t0": round(a, 1), "t1": round(b, 1), "origin": "telemetry"} for a, b in merged]
    cue_t: list[float] = []
    for t, w, _ in sorted(inp.hits):
        if w < 0.8 or any(abs(t - c) < 45 for c in cue_t):
            continue
        cue_t.append(t)
        # Words explained by a telemetry event: 90 s after a weak one, 5 min after
        # a strong one (a big crash is talked about for a while).
        explained = False
        for c in cands:
            if c["origin"] != "telemetry" or not c["t0"] - 1 <= t <= c["t1"] + 300:
                continue
            s = c.setdefault("_tel", telemetry_features(inp, c["t0"], c["t1"], p)["score"])
            if s >= 0.5 or (s >= 0.15 and t <= c["t1"] + 90):
                explained = True
                break
        if explained:
            continue
        # The fall is just before the words: the strongest keyframe in 20 s.
        tc = t - 5.0
        if inp.kf_crash is not None:
            vt = inp.visual.t.astype(float)
            sel = (vt >= t - 20.0) & (vt <= t - 1.0)
            if sel.any():
                tc = float(vt[sel][int(np.argmax(inp.kf_crash[sel]))])
        cands.append(
            {
                "t0": round(max(0.0, tc - 1.0), 1),
                "t1": round(tc + 1.0, 1),
                "origin": "words",
                "cue_t": round(t, 1),
            }
        )
    for c in cands:
        c.pop("_tel", None)
    return cands


def score_candidate(inp: SourceInputs, c: dict[str, Any], p: CrashParams) -> dict[str, Any]:
    """Telemetry, visual, audio and transcript sub-scores plus the pre-score."""
    t0, t1 = c["t0"], c["t1"]
    tel = telemetry_features(inp, t0, t1, p) if inp.imu is not None else {"score": 0.0}
    vis = visual_feature(inp, t0, t1)
    aud = audio_feature(inp, t0, t1)
    wrd = words_feature(inp, t0, t1)
    w = dict(p.weights)
    scores = {
        "telemetry": tel["score"],
        "visual": vis["score"],
        "audio": aud["score"],
        "words": wrd["score"],
    }
    pre = _fuse(scores, w, ("telemetry", "visual", "audio", "words"))
    return {
        **c,
        "source": str(inp.path),
        "name": inp.name,
        "scores": scores,
        "pre_score": round(pre, 3),
        "telemetry": tel,
        "visual": vis,
        "audio": aud,
        "words": wrd,
    }


def _chain(words: list[dict[str, Any]], i: int, gap: float = 0.8) -> tuple[int, int]:
    """Indices of the first and last word of the utterance chain around word ``i``."""
    a = b = i
    while a > 0 and words[a]["start"] - words[a - 1]["end"] < gap:
        a -= 1
    while b + 1 < len(words) and words[b + 1]["start"] - words[b]["end"] < gap:
        b += 1
    return a, b


def _normal_view(inp: SourceInputs, t0: float, t1: float) -> bool:
    """True when the keyframes of ``[t0, t1]`` no longer look like a crash (the
    rider is up: not lying in the snow looking at the sky)."""
    if inp.kf_crash is None:
        return True
    t = inp.visual.t.astype(float)
    sel = (t >= t0) & (t <= t1)
    return bool(not sel.any() or float(np.median(inp.kf_crash[sel])) < 0.6)


def plan_clip(inp: SourceInputs, c: dict[str, Any], p: CrashParams) -> dict[str, Any]:
    """Clip for a crash: the approach, the impact and the aftermath.

    Starts ``lead_s`` before the first shock. After the last one it runs at
    least ``min_after_s``; if the rider stopped, it plays on until they move
    again or the first thing said after the stop has been said, capped at
    ``max_after_s`` after the last shock and ``max_clip_s`` in total. A
    ``reaction`` is planned too: one utterance (at most 12 s) with the
    strongest crash words of the next 4 minutes, preferring one said once the
    view is back to normal (a rider lying in the snow films the sky for
    minutes while talking).
    """
    t0, t1 = c["t0"], c["t1"]
    start = max(0.0, t0 - p.lead_s)
    hard_end = min(inp.duration, t1 + p.max_after_s, start + p.max_clip_s)
    end = min(hard_end, t1 + p.min_after_s)
    speed = _grid(inp, "speed_kmh")
    words = inp.words or []
    why = "min aftermath"
    sp = _seg(speed, t1, t1 + 12.0)
    if speed is not None and sp.size and sp.min() <= p.still_kmh + 1.0:
        i_stop = int(t1 / DT) + int(np.argmax(sp <= p.still_kmh + 1.0))
        t_stop = i_stop * DT
        x = np.nan_to_num(speed[i_stop : int(hard_end / DT)], nan=0.0)
        go = np.convolve(x > 5.0, np.ones(20), mode="valid") >= 20  # moving for 2 s
        t_go = t_stop + float(np.argmax(go)) * DT if go.any() else hard_end
        after = [k for k, wd in enumerate(words) if t_stop <= wd["start"] < hard_end]
        t_talk = words[_chain(words, after[0])[1]]["end"] + 0.5 if after else hard_end
        cand = min(t_go, t_talk, hard_end)
        if cand > end:
            end, why = cand, ("moving again" if t_go <= t_talk else "first words after")
    clip = {"start": round(start, 2), "end": round(end, 2), "until": why}
    best = None  # (view is back to normal, weight, word index): best wins, earliest on ties
    for k, wd in enumerate(words):
        if not end + 2.0 < wd["start"] <= end + 240.0:
            continue
        hit = [w for t, w, _ in inp.hits if abs(t - wd["start"]) < 0.05]
        if not hit or max(hit) < 0.9:
            continue
        a, b = _chain(words, k)
        rank = (_normal_view(inp, words[a]["start"], words[b]["end"] + 3.0), max(hit), k)
        if best is None or rank[:2] > best[:2]:
            best = rank
    if best is not None:
        best = (best[1], best[2])
        a, b = _chain(words, best[1])
        r0 = max(end, words[a]["start"] - 0.4)
        r1 = min(words[b]["end"] + 1.5, r0 + 12.0, inp.duration)
        clip["reaction"] = {
            "start": round(r0, 2),
            "end": round(r1, 2),
            "text": " ".join(w["word"].strip() for w in words[a : b + 1])[:200],
        }
    return clip


# --------------------------------------------------------------------------- dense step
def _dense_cache(src: Path, p: CrashParams) -> Path:
    return cache.source_dir(src) / f"crashdense-{p.dense_key()}.json"


def dense_window(c: dict[str, Any], p: CrashParams) -> tuple[float, float]:
    """Decode range of the confirm step (at most 24 s)."""
    a = max(0.0, c["t0"] - p.dense_pre_s)
    b = min(c["t1"] + p.dense_post_s, a + 24.0)
    return round(a, 1), round(b, 1)


def dense_feature(
    enc: ClipEncoder | None,
    inp: SourceInputs,
    c: dict[str, Any],
    p: CrashParams,
    store: dict[str, Any],
) -> dict[str, Any] | None:
    """Dense crash-prompt and view-change evidence (cached in ``store``)."""
    from cutpoint.config import remap_source  # noqa: PLC0415

    a, b = dense_window(c, p)
    key = f"{a:.1f}-{b:.1f}"
    rec = store.get(key)
    if rec is None:
        if enc is None:
            return None
        ts, frames = decode_window(remap_source(inp.path), a, b, p.dense_fps)
        if not len(ts):
            return None
        emb = enc.embed(frames)
        rec = {
            "t": [round(float(x), 2) for x in ts],
            "crash": [round(float(x), 3) for x in crash_prompt_scores(emb, p)],
            "change": [round(float(x), 3) for x in view_change(emb)],
        }
        store[key] = rec
    t = np.asarray(rec["t"])
    cs, ch = np.asarray(rec["crash"]), np.asarray(rec["change"])
    at = (t >= c["t0"] - 0.5) & (t <= c["t1"] + 3.0)
    hit = (t >= c["t0"] - 0.5) & (t <= c["t1"] + 1.0)
    lead = t < c["t0"] - 1.0
    if not at.any():
        return None
    q = np.sort(cs[at])[::-1]
    top = float(q[: max(1, len(q) // 4)].mean())
    chaos = float((ch[hit] >= 0.1).mean()) if hit.any() else 0.0
    chaos_lead = float((ch[lead] >= 0.1).mean()) if lead.any() else 0.0
    s = 0.5 * _clip01((top - 0.45) / 0.4) + 0.5 * _clip01((chaos - chaos_lead) / 0.25)
    return {
        "score": round(s, 3),
        "crash_topq": round(top, 3),
        "chaos": round(chaos, 3),
        "chaos_lead": round(chaos_lead, 3),
        "frames": len(t),
        "window": [a, b],
    }


# --------------------------------------------------------------------------- sheets
def write_sheet(
    c: dict[str, Any],
    out: Path,
    fps: float = 4.0,
    pre_s: float = 4.0,
    post_s: float = 6.0,
    cols: int = 8,
    width: int = 240,
    span: tuple[float, float] | None = None,
) -> Path | None:
    """Contact sheet of a candidate: ``pre_s`` before the first shock to ``post_s``
    after the last (or ``span``), ``fps`` frames per second, captioned with the time."""
    import cv2  # noqa: PLC0415

    from cutpoint.config import remap_source  # noqa: PLC0415

    a, b = span if span else (max(0.0, c["t0"] - pre_s), c["t1"] + post_s)
    ts, frames = decode_window(remap_source(c["source"]), a, b, fps, width)
    if not len(ts):
        return None
    tiles = []
    for t, fr in zip(ts, frames, strict=True):
        img = cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        rel = t - c["t0"]
        hot = c["t0"] - 0.05 <= t <= c["t1"] + 0.05
        cv2.rectangle(img, (0, h - 18), (w, h), (0, 0, 160) if hot else (0, 0, 0), -1)
        cv2.putText(
            img,
            f"{t:.1f}s ({rel:+.1f})",
            (3, h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(img)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    title = np.zeros((26, grid.shape[1], 3), np.uint8)
    cv2.putText(
        title,
        c.get("caption", "")[:150],
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([title, grid]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return out


# --------------------------------------------------------------------------- project
def _project_sources(project: Path) -> list[tuple[Path, float]]:
    from cutpoint.events.common import load_highlights  # noqa: PLC0415

    hl = load_highlights(project)
    seen: dict[str, float] = {}
    for s in hl["sources"]:  # recovered parts share the path
        seen.setdefault(s["path"], float(s["duration"]))
    return [(Path(k), v) for k, v in seen.items()]


def _confidence(c: dict[str, Any], w: dict[str, float]) -> float:
    """Weighted fusion of the sub-scores.

    The visual and dense evidence is halved when the rider was not moving
    before the event: an upside-down, spinning view while standing is the
    camera being taken off or handled, not a fall.
    """
    s = dict(c["scores"])
    moving = (c.get("telemetry") or {}).get("moving_before")
    if moving is not None:
        for k in ("visual", "dense"):
            if k in s:
                s[k] *= 0.5 + 0.5 * moving
    use = tuple(k for k in ("telemetry", "visual", "dense", "audio", "words") if k in s)
    return _fuse(s, w, use)


def _absorb(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop candidates inside a stronger candidate's clip (same source).

    The lead-in and the aftermath of a crash (lying in the snow, getting up)
    trigger their own weaker candidates; they are listed under ``absorbed``.
    """
    kept: list[dict[str, Any]] = []
    for c in cands:  # strongest first
        host = next(
            (
                k
                for k in kept
                if k["source"] == c["source"]
                and k["clip"]["start"] - 1.0 <= c["t0"] <= k["clip"]["end"] + 1.0
            ),
            None,
        )
        if host is None:
            kept.append(c)
        else:
            host.setdefault("absorbed", []).append(
                {"t0": c["t0"], "t1": c["t1"], "confidence": c["confidence"], "origin": c["origin"]}
            )
    return kept


def load_crashes(project: Path) -> dict[str, Any] | None:
    """``<project>/crashes.json`` or None."""
    return cache.load_json(project / CRASHES_FILE)


def detect_project(
    project: Path,
    params: CrashParams | None = None,
    *,
    dense: bool = True,
    sheets: int = 15,
    force: bool = False,
    tmp_dir: Path | None = None,
    activity_name: str | None = None,
    context: str | None = None,
    refresh_profile: bool = False,
) -> Path:
    """Find crashes in every source of a project and write ``crashes.json``.

    Needs ``cutpoint analyze`` (audio, transcript, telemetry); the visual and voice
    layers (``cutpoint events``) are used when cached. Only the IMU grids and the
    dense confirm decode anything; both are cached.

    Args:
        project: Project folder (highlights.json).
        params: Thresholds / weights / prompts.
        dense: Run the dense confirm on the best candidates (GPU, ~2 s each).
        sheets: Contact sheets for this many top candidates
            (``<project>/crash_check``); 0 = none.
        force: Recompute the IMU grids and the dense step.
        tmp_dir: Scratch folder for GPMF-only remuxes of files on slow storage.
        activity_name: Activity profile by name (default: detected, see
            :func:`cutpoint.activity.resolve`); ignored when ``params`` brings
            its own prompts or words.
        context: The user's words about the video (refines the profile).
        refresh_profile: Re-detect / re-synthesize the profile.

    Returns:
        Path of crashes.json.
    """
    sources = []
    for src, dur in _project_sources(project):
        try:
            cache.source_dir(src)  # every cached layer is keyed by the file's size and mtime
        except OSError as err:
            logger.warning("%s: footage not reachable (%s); skipped", src, err)
            continue
        sources.append((src, dur))
    if not sources:
        raise ValueError(f"No footage of {project.name} is reachable; reconnect it and rerun")
    p = params or CrashParams()
    prof = None
    if (p.pos, p.neg, p.words) == (CRASH_POS, CRASH_NEG, tuple(CRASH_WORDS.items())):
        prof = activity.resolve(
            project, activity=activity_name, context=context, refresh=refresh_profile
        )
        p = p.with_profile(prof["profile"])
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    inputs: dict[str, SourceInputs] = {}
    t0 = time.perf_counter()
    for src, dur in sources:
        if force:
            imu_grids(src, dur, force=True, tmp_dir=tmp_dir)
        inputs[str(src)] = load_inputs(src, dur, p, tmp_dir)
    timings["inputs_s"] = round(time.perf_counter() - t0, 1)
    cands = []
    for inp in inputs.values():
        cands += [score_candidate(inp, c, p) for c in candidate_events(inp, p)]
    cands.sort(key=lambda c: -c["pre_score"])
    w = dict(p.weights)
    t0 = time.perf_counter()
    todo = [c for c in cands if c["pre_score"] >= p.prefilter][: p.dense_top] if dense else []
    enc = None
    stores: dict[str, dict[str, Any]] = {}
    try:
        for c in todo:
            src = c["source"]
            if src not in stores:
                stores[src] = {} if force else (cache.load_json(_dense_cache(Path(src), p)) or {})
            a, b = dense_window(c, p)
            if enc is None and f"{a:.1f}-{b:.1f}" not in stores[src]:
                enc = ClipEncoder()
            c["dense"] = dense_feature(enc, inputs[src], c, p, stores[src])
    finally:
        if enc is not None:
            enc.close()
    for src, store in stores.items():
        cache.save_json(_dense_cache(Path(src), p), store)
    timings["dense_s"] = round(time.perf_counter() - t0, 1)
    for c in cands:
        d = c.get("dense")
        if d is not None:
            c["scores"]["dense"] = d["score"]
        c["confidence"] = round(_confidence(c, w), 3)
        c["confirmed"] = d is not None
        c["clip"] = plan_clip(inputs[c["source"]], c, p)
    cands = _absorb(sorted(cands, key=lambda c: -c["confidence"]))
    for k, c in enumerate(cands, 1):
        c["id"] = f"c{k:03d}"
    t0 = time.perf_counter()
    written = []
    out_dir = project / CHECK_DIR
    if sheets:
        for f in out_dir.glob("*.jpg"):
            f.unlink()
    for c in cands[:sheets]:
        c["caption"] = (
            f"{c['id']} {c['name']} {c['t0']:.1f}-{c['t1']:.1f}s "
            f"conf {c['confidence']:.2f} tel {c['scores']['telemetry']:.2f} "
            f"vis {c['scores']['visual']:.2f} "
            f"dense {c['scores'].get('dense', 0):.2f} "
            f"aud {c['scores']['audio']:.2f} words {c['scores']['words']:.2f}"
        )
        path = write_sheet(c, out_dir / f"{c['id']}_{Path(c['name']).stem}_{c['t0']:.0f}.jpg")
        if path:
            c["sheet"] = str(path)
            written.append(path)
    timings["sheets_s"] = round(time.perf_counter() - t0, 1)
    timings["total_s"] = round(time.perf_counter() - t_start, 1)
    data = {
        "version": CRASH_VERSION,
        "activity": prof and {k: prof.get(k) for k in ("name", "source")},
        "params": {**asdict(p), "key": p.key()},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timings": timings,
        "n_candidates": len(cands),
        "crashes": cands,
    }
    out = project / CRASHES_FILE
    cache.save_json(out, data)
    for c in cands[:10]:
        logger.info(
            "%s %s %6.1f-%6.1fs conf %.2f (tel %.2f vis %.2f dense %s aud %.2f words %.2f)",
            c["id"],
            c["name"],
            c["t0"],
            c["t1"],
            c["confidence"],
            c["scores"]["telemetry"],
            c["scores"]["visual"],
            f"{c['scores']['dense']:.2f}" if "dense" in c["scores"] else "-",
            c["scores"]["audio"],
            c["scores"]["words"],
        )
    logger.info(
        "Wrote %s (%d candidates, %d confirmed densely, %d sheets) in %.1fs",
        out,
        len(cands),
        sum(c["confirmed"] for c in cands),
        len(written),
        timings["total_s"],
    )
    return out
