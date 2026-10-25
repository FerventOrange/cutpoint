"""Ski / snowboard scoring: telemetry + audio + VLM features, masks and clip picking.

Used by ``cutpoint analyze --style ski`` (and learned presets based on it) and by
``cutpoint refmatch`` to evaluate and fit weights. Every feature is 0-1 on the 0.1 s
analysis grid of one source:

    airtime  telemetry airtime events (strength by duration and confidence)
    crash    telemetry crash events (impact g followed by a stop; head tilt raises it)
    vlm      ``cutpoint events`` window tags: max of jump_air, landing, powder_spray
             and friends_faces (0-10 -> 0-1); friends only outside the lift/still
             mask, scaled down by the window's chairlift / standing_talking /
             lodge tags; 0 before ``cutpoint events`` has run
    speed    GPS speed percentile within the day (above-median part, 0-1); the
             GoPro wind meter stands in where there is no GPS fix
    social   PANNs Laughter / Scream / Cheering (after ``cutpoint events``) and
             transcript laughter
    carve    telemetry turn rhythm x speed
    scenic   VLM tag scenic_vista
    speech   capped style keywords (kept low: wind and chatter are constant)
    faces    GoPro FACE detections (weight 0 by default; available to fitting)
    stopped  1 while stopped (telemetry still_mask); weight 0 by default, only
             useful with a soft mask (``mask_friends`` 0, learned presets)
    short_clip 1 for a whole clip shorter than 30 s (recorded on purpose);
             weight 0 by default

Features from the ``cutpoint events`` voice and visual layers (0 until they ran):

    beauty   beauty-shot score: CLIP zero-shot vista-vs-junk contrast x LAION
             aesthetic head x steadiness (telemetry speed; embedding change
             without GPS)
    people   friends close to the camera: Faster R-CNN person boxes (the
             wearer's own legs/skis dropped) by closeness and count, boosted by
             the zero-shot group prompt
    group    the zero-shot "friends / group of people" prompt alone
    (360 clips: the same three from :mod:`cutpoint.pano.views`, taking the
    best of 8 levelled horizon views per second, wearer excluded)
    laughter PANNs laughter on 200 Hz high-passed audio above a wind-raised
             threshold (wind from the low-band energy and telemetry speed)
    voice    Silero VAD and PANNs speech both above wind-raised thresholds
    taste    personal taste head (``cutpoint refmatch --taste``): logistic regression
             on the frame embeddings, fitted on a reference edit

From ``cutpoint crashes`` (``<project>/crashes.json``, 0 until it ran):

    wipeout  crash confidence (telemetry + visual + dense confirm + audio +
             transcript, :mod:`cutpoint.crash`) from the first shock to
             3 s after the last. Crashes at or above the style's
             ``must_include_crash`` also become must-include candidates
             (:func:`must_include_picks`), whatever the weights say.

The weighted sum is hard-masked on lifts and stops (``lift_mask | still_mask``);
friends in view do not lift the mask. Peaks are
turned into clips whose length depends on the dominant feature: jumps 3-6 s
(1.5 s before take-off to 1.5 s after landing), crashes 4-8 s, speed/carve
4-10 s, scenic 2-4 s, social/VLM 4-8 s. Each pick takes its whole stretch
(the run around the peak above ``STRETCH`` of it), so a flat stretch gives
one clip, not several back to back.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.config import path_key, source_path
from cutpoint.telemetry import DT, Telemetry, extract, join_runs

logger = logging.getLogger(__name__)

FEATURES = (
    "airtime",
    "crash",
    "vlm",
    "speed",
    "social",
    "carve",
    "scenic",
    "speech",
    "faces",
    "stopped",
    "short_clip",
    "beauty",
    "people",
    "group",
    "laughter",
    "voice",
    "taste",
    "wipeout",
)
# Features computed from the voice / visual layers and the taste head.
NEW_FEATURES = ("beauty", "people", "group", "laughter", "voice", "taste")
SHORT_CLIP_S = 30.0  # a clip this short was recorded on purpose for one moment
VLM_ACTION_TAGS = ("jump_air", "landing", "powder_spray")
VLM_IDLE_TAGS = ("chairlift", "standing_talking", "lodge")  # friends do not count there
FRIENDS_STRONG = 7  # default mask_friends: > 0 = hard lift/still mask, 0 = soft
STRETCH = 0.7  # a pick takes the run around its peak that stays above this share of it
FEATURES_DIR = "ski_features"
# Clip geometry per dominant feature: (min_len, max_len, pre, post) around the peak.
GEOMETRY: dict[str, tuple[float, float, float, float]] = {
    "airtime": (3.0, 6.0, 1.5, 1.5),  # pre/post relative to take-off / landing
    "crash": (4.0, 8.0, 2.0, 4.0),  # relative to the impact
    "speed": (4.0, 10.0, 3.0, 3.0),
    "carve": (4.0, 10.0, 3.0, 3.0),
    "scenic": (2.0, 4.0, 1.5, 1.5),
    "social": (4.0, 8.0, 3.5, 2.5),
    "vlm": (4.0, 8.0, 3.0, 3.0),
    "speech": (4.0, 8.0, 3.0, 2.5),
    "faces": (3.0, 6.0, 2.0, 2.0),
    "stopped": (3.0, 6.0, 2.5, 2.5),
    "short_clip": (2.0, 8.0, 4.0, 4.0),
    "beauty": (2.0, 4.0, 1.5, 1.5),
    "people": (3.0, 6.0, 2.0, 2.0),
    "group": (3.0, 6.0, 2.0, 2.0),
    "laughter": (4.0, 8.0, 3.5, 2.5),
    "voice": (4.0, 8.0, 3.0, 2.5),
    "taste": (3.0, 6.0, 2.0, 2.0),
    "wipeout": (6.0, 40.0, 5.0, 8.0),  # crashes.json plans its own clip
}
KIND_LABEL = {
    "airtime": "jump",
    "crash": "crash",
    "speed": "fast",
    "carve": "carve",
    "scenic": "scenic",
    "social": "laugh",
    "vlm": "vlm",
    "speech": "talk",
    "faces": "friends",
    "stopped": "stop",
    "short_clip": "clip",
    "beauty": "view",
    "people": "friends",
    "group": "crew",
    "laughter": "laugh",
    "voice": "talk",
    "taste": "pick",
    "wipeout": "crash",
}


@dataclass
class SkiContext:
    """Day-level inputs shared by every source of a ski analysis."""

    project: Path
    telemetry: dict[str, Telemetry | None]
    speed_ref: np.ndarray  # sorted speeds (km/h) of the day's moving samples
    audio_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    windows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    runs: dict[str, Any] = field(default_factory=dict)
    taste: dict[str, Any] | None = None  # {"w": (768,), "b": float} of a learned preset
    crashes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # crashes.json


def _key(path: str | Path) -> str:
    return path_key(str(Path(path)))


def prepare(
    sources: list[Path], durations: dict[str, float], project: Path, taste: Path | None = None
) -> SkiContext:
    """Extract telemetry for every source, load events.json, write telemetry.json.

    Args:
        sources: Source videos (original paths, as recorded in highlights.json).
        durations: ``{str(source): container duration}``.
        project: Project folder.
        taste: Taste head (``.npz`` with ``w``, ``b``) of a learned preset.
    """
    tels: dict[str, Telemetry | None] = {}
    for src in sources:
        tels[str(src)] = extract(src, durations[str(src)])
    good = [t for t in tels.values() if t is not None]
    speeds = []
    for t in good:
        g = t.grid
        moving = (g["gps_ok"] > 0) & (g["lift_mask"] == 0) & (g["still_mask"] == 0)
        speeds.append(g["speed_kmh"][moving & (g["speed_kmh"] > 5)])
    speed_ref = np.sort(np.concatenate(speeds)) if speeds else np.zeros(0)
    runs = join_runs(good) if good else {}
    ctx = SkiContext(project=project, telemetry=tels, speed_ref=speed_ref, runs=runs)
    if taste is not None and Path(taste).exists():
        ctx.taste = load_taste(Path(taste))
    ev = cache.load_json(project / "events.json")
    if ev:
        for e in ev.get("events", []):
            if e.get("kind") == "audio":
                ctx.audio_events.setdefault(_key(e["source"]), []).append(e)
        for w in ev.get("windows", []):
            ctx.windows.setdefault(_key(w["source"]), []).append(w)
        logger.info(
            "ski: using events.json (%d audio events, %d VLM windows)",
            sum(map(len, ctx.audio_events.values())),
            sum(map(len, ctx.windows.values())),
        )
    from cutpoint.crash import load_crashes  # noqa: PLC0415

    cr = load_crashes(project)
    if cr:
        for c in cr.get("crashes", []):
            ctx.crashes.setdefault(_key(c["source"]), []).append(c)
        top = max((c["confidence"] for c in cr.get("crashes", [])), default=0.0)
        logger.info(
            "ski: using crashes.json (%d candidates, best confidence %.2f)",
            len(cr.get("crashes", [])),
            top,
        )
    write_telemetry_json(project, ctx)
    return ctx


def write_telemetry_json(project: Path, ctx: SkiContext) -> Path:
    """``<project>/telemetry.json``: day summary, runs, per-source summary and events."""
    data = {
        "version": 1,
        "day": ctx.runs.get("day"),
        "runs": ctx.runs.get("runs", []),
        "clips": ctx.runs.get("clips", []),
        "sources": {
            src: ({**t.summary(), "events": t.events} if t else None)
            for src, t in ctx.telemetry.items()
        },
    }
    path = project / "telemetry.json"
    cache.save_json(path, data)
    day = data["day"] or {}
    if day:
        logger.info(
            "telemetry: %d clips, top %.0f km/h, %d runs, %d airtime (%d high), "
            "%d crash, lift %.0f s, stopped %.0f s",
            day["clips"],
            day["top_speed_kmh"] or 0,
            day["runs"],
            day["airtime_events"],
            day["airtime_high"],
            day["crash_events"],
            day["lift_s"],
            day["still_s"],
        )
    return path


# --------------------------------------------------------------------------- features
def _paint(n: int, spans: list[tuple[float, float, float]]) -> np.ndarray:
    x = np.zeros(n)
    for a, b, v in spans:
        i, j = max(0, int(a / DT)), min(n, int(b / DT) + 1)
        if i < j:
            x[i:j] = np.maximum(x[i:j], v)
    return x


def _resize(x: np.ndarray, n: int, fill: float = 0.0) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.full(n - len(x), fill)])


def build_features(
    src: Path, n: int, sig: dict[str, np.ndarray], ctx: SkiContext, offset: float = 0.0
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Feature grids, the hard mask and the friends-exception grid for one source.

    Args:
        src: Source path (key into the context).
        n: Grid length of this (part of the) source.
        sig: Generic analysis signals (``speech``, ``laughter``) on the same grid.
        ctx: Day context.
        offset: Start of a recovered part (grids are sliced).

    Returns:
        ``(features, mask, friends)``; ``mask`` is 1 where clips are not allowed.
    """
    tel = ctx.telemetry.get(str(src))
    o = int(offset / DT)
    f = {k: np.zeros(n) for k in FEATURES}
    mask = np.zeros(n)
    sp_kmh = None
    if tel is not None:
        g = {k: _resize(v[o:], n) for k, v in tel.grid.items()}
        f["airtime"], f["crash"], f["carve"] = g["airtime"], g["crash"], g["carve"]
        # Without GPS the wind meter stands in for speed, but only while the IMU
        # shows riding vibration (handling the camera indoors also moves the meter).
        proxy = 55.0 * g["wind"] * (g["vibration"] >= 0.12) * (g["indoor"] < 0.5)
        sp = np.where(g["gps_ok"] > 0, np.nan_to_num(g["speed_kmh"]), proxy)
        sp_kmh = sp
        if ctx.speed_ref.size:
            pct = np.searchsorted(ctx.speed_ref, sp) / ctx.speed_ref.size
            f["speed"] = np.clip((pct - 0.5) / 0.5, 0, 1)
        f["faces"] = np.clip(g["faces"] / 0.02, 0, 1)
        f["stopped"] = g["still_mask"].copy()
        mask = np.maximum(g["lift_mask"], g["still_mask"])
        f["speed"] *= 1 - mask
        f["carve"] *= 1 - mask
    if tel is not None and tel.duration <= SHORT_CLIP_S:
        f["short_clip"][:] = 1.0
    f["speech"] = sig.get("speech", np.zeros(n))[:n]
    social = sig.get("laughter", np.zeros(n))[:n].copy()
    norm = {"Laughter": 0.12, "Scream": 0.3, "Cheering": 0.3}
    spans = [
        (
            e["start"] - offset - 0.5,
            e["end"] - offset + 0.5,
            min(1.0, (e.get("confidence") or 0) / norm[e["label"]]),
        )
        for e in ctx.audio_events.get(_key(src), [])
        if e.get("label") in norm
    ]
    f["social"] = np.maximum(social, _paint(n, spans))
    friends = np.zeros(n)
    wins = ctx.windows.get(_key(src), [])
    if wins:
        act, scen, fr, fr_ride = [], [], [], []
        for w in wins:
            tags = w.get("tags") or {}
            a, b = w["start"] - offset, w["end"] - offset
            act.append((a, b, max(tags.get(t, 0) for t in VLM_ACTION_TAGS) / 10))
            scen.append((a, b, tags.get("scenic_vista", 0) / 10))
            fr.append((a, b, tags.get("friends_faces", 0) / 10))
            # Friends raise a riding moment, not lift, standing or lodge time, where
            # they are in view for minutes: scaled down by the window's idle tags.
            idle = max(tags.get(t, 0) for t in VLM_IDLE_TAGS) / 10
            fr_ride.append((a, b, tags.get("friends_faces", 0) / 10 * (1 - idle)))
        f["scenic"], friends = _paint(n, scen), _paint(n, fr)
        # ... and only outside the telemetry lift/still mask.
        f["vlm"] = np.maximum(_paint(n, act), _paint(n, fr_ride) * (1 - mask))
    f["wipeout"] = _paint(
        n,
        [
            (c["t0"] - offset, c["t1"] - offset + 3.0, c["confidence"])
            for c in ctx.crashes.get(_key(src), [])
        ],
    )
    vis = layer_features(src, n, offset, sp_kmh, ctx.taste)
    f.update(vis)
    f["friends_vlm"] = friends.copy()  # not a scored feature: kept for ablations
    friends = np.maximum(friends, vis.get("people", np.zeros(n)))
    return f, mask, friends


def layer_features(
    src: Path,
    n: int,
    offset: float = 0.0,
    speed_kmh: np.ndarray | None = None,
    taste: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Features from the cached voice / visual layers (missing layers are skipped).

    Generic: nothing here needs telemetry. ``speed_kmh`` (on the grid) only
    sharpens the steadiness modifier of ``beauty`` and the wind level of the
    voice features.

    Args:
        src: Source path.
        n: Grid length of this (part of the) source.
        offset: Start of a recovered part.
        speed_kmh: Optional speed (km/h) on the same grid.
        taste: Optional taste head ``{"w", "b"}``.
    """
    from cutpoint.events import visual as V  # noqa: PLC0415
    from cutpoint.events import voice as VO  # noqa: PLC0415

    out: dict[str, np.ndarray] = {}
    o = round(offset / DT)
    vg = VO.load_voice(src)
    if vg is not None:
        moving = None if speed_kmh is None else np.clip(speed_kmh / 40.0, 0, 1)
        grids = {k: _resize(v[o:], n) for k, v in vg.items()}
        vf = VO.voice_features(grids, moving)
        out["laughter"], out["voice"] = vf["laughter"], vf["voice"]
    from cutpoint.pano.views import layer_grids  # noqa: PLC0415

    pano = layer_grids(src, n, offset, DT)  # 360 clip: best direction per second
    if pano is not None:
        out.update(pano)
    vd = V.load_visual(src)
    if vd is not None and len(vd.t):
        t = vd.t.astype(float) - offset
        sc = V.frame_scores(vd)
        steady = None
        if speed_kmh is not None:  # telemetry speed at the frame times
            idx = np.clip(np.round(t / DT).astype(int), 0, n - 1)
            steady = np.clip(1 - speed_kmh[idx] / 35.0, 0, 1)
        grid = (
            {}
            if pano is not None
            else {
                "beauty": V.beauty_score(sc, steady),
                "people": V.friends_score(sc),
                "group": sc["group_zs"],
            }
        )
        if taste is not None:
            grid["taste"] = taste_frames(vd, taste)
        for k, v in grid.items():
            out[k] = V.to_grid(t, smooth3(v), n, DT)
    return out


def smooth3(x: np.ndarray) -> np.ndarray:
    """Median of each frame and its neighbours (drops one-frame flickers)."""
    if len(x) < 3:
        return x
    p = np.r_[x[0], x, x[-1]]
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), 0)


def load_taste(path: Path) -> dict[str, Any]:
    """Taste head saved by ``cutpoint refmatch --taste`` (``w`` (768,), ``b``)."""
    with np.load(path) as z:
        return {"w": z["w"].astype(np.float32), "b": float(z["b"]), "path": str(path)}


def taste_frames(vd: Any, taste: dict[str, Any]) -> np.ndarray:
    """Taste probability per frame (mean crop embedding through the logistic head)."""
    z = vd.mean_emb @ taste["w"] + taste["b"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def effective_mask(
    mask: np.ndarray, friends: np.ndarray, friends_strong: float = FRIENDS_STRONG
) -> np.ndarray:
    """1 where clips are forbidden: lift/stop.

    ``friends_strong`` (a style's ``mask_friends``) above 0 keeps the mask
    hard; 0 or less means a *soft* mask (nothing is forbidden; the
    ``stopped`` weight decides). Friends no longer lift the mask: people
    standing and chatting, or on the chairlift, are in view for minutes, so
    ``friends`` is accepted for the callers but not used.
    """
    if friends_strong <= 0:
        return np.zeros_like(mask)
    return mask.copy()


def combine(
    features: dict[str, np.ndarray],
    weights: dict[str, float],
    mask: np.ndarray,
    friends: np.ndarray,
    friends_strong: float = FRIENDS_STRONG,
) -> np.ndarray:
    """Weighted feature sum with the hard mask applied."""
    n = len(mask)
    total = np.zeros(n)
    for k, w in weights.items():
        if k in features and w:
            total += w * features[k][:n]
    return total * (1 - effective_mask(mask, friends, friends_strong))


# --------------------------------------------------------------------------- picking
@dataclass
class Pick:
    """A clip chosen from the combined curve (part-relative times)."""

    start: float
    end: float
    peak_t: float
    peak: float
    kind: str
    signals: dict[str, float]
    label: str
    event: dict[str, Any] | None = None


def _nearest_event(
    events: list[dict[str, Any]], typ: str, t: float, slack: float
) -> dict[str, Any] | None:
    best = None
    for e in events:
        if e["type"] != typ or not (e["t"] - slack <= t <= e["end"] + slack):
            continue
        if best is None or abs((e["t"] + e["end"]) / 2 - t) < abs(
            (best["t"] + best["end"]) / 2 - t
        ):
            best = e
    return best


def pick(
    features: dict[str, np.ndarray],
    weights: dict[str, float],
    mask: np.ndarray,
    friends: np.ndarray,
    events: list[dict[str, Any]],
    duration: float,
    max_picks: int,
    offset: float = 0.0,
    friends_strong: float = FRIENDS_STRONG,
) -> list[Pick]:
    """Peak-pick clips with feature-specific geometry; results do not overlap.

    Args:
        features, weights, mask, friends: As for :func:`combine`.
        events: Telemetry events in source time (shifted by ``-offset``).
        duration: Length of this (part of the) source.
        max_picks: Upper bound on clips.
        offset: Start of a recovered part.
        friends_strong: A style's ``mask_friends`` (see :func:`effective_mask`).
    """
    combined = combine(features, weights, mask, friends, friends_strong)
    n = len(combined)
    score = combined.copy()
    gmax = float(score.max()) if n else 0.0
    evs = [{**e, "t": e["t"] - offset, "end": e["end"] - offset} for e in events]
    picks: list[Pick] = []
    taken = np.zeros(n, dtype=bool)  # suppressed by a pick: its clip and its stretch
    while len(picks) < max_picks and gmax > 0:
        i = int(score.argmax())
        peak = float(score[i])
        if peak <= 0.02 or peak < 0.08 * gmax:
            break
        # The stretch: the run around the peak that stays above STRETCH of it (a
        # flat VLM window, a long fast section) is one moment and gives one clip.
        # A stretch that reaches an earlier pick's, with no dip below STRETCH
        # between them, is that moment's shoulder: suppressed, not picked.
        s0, s1 = i, i
        while s0 > 0 and combined[s0 - 1] >= STRETCH * peak:
            s0 -= 1
        while s1 < n - 1 and combined[s1 + 1] >= STRETCH * peak:
            s1 += 1
        if taken[max(0, s0 - 1) : s1 + 2].any():
            score[s0 : s1 + 1] = 0.0
            taken[s0 : s1 + 1] = True
            continue
        # On a flat top, centre on the run of samples at the peak, not its first sample.
        r0, r1 = i, i
        while r0 > s0 and combined[r0 - 1] >= peak:
            r0 -= 1
        while r1 < s1 and combined[r1 + 1] >= peak:
            r1 += 1
        i = (r0 + r1) // 2
        t = i * DT
        contrib = {k: weights.get(k, 0) * features[k][i] for k in features}
        kind = max(contrib, key=contrib.get)
        lo_len, hi_len, pre, post = GEOMETRY.get(kind, GEOMETRY["vlm"])
        ev = None
        if kind == "airtime" and (ev := _nearest_event(evs, "airtime", t, 0.5)):
            a, b = ev["t"] - pre, ev["end"] + post
        elif kind == "crash" and (ev := _nearest_event(evs, "crash", t, 1.0)):
            tc = ev.get("impact_t", ev["t"]) - offset if "impact_t" in ev else ev["t"]
            a, b = tc - pre, tc + post
        elif kind in ("speed", "carve"):
            j0, j1 = i, i
            while j0 > 0 and combined[j0 - 1] >= 0.75 * peak and (i - j0) * DT < hi_len / 2:
                j0 -= 1
            while j1 < n - 1 and combined[j1 + 1] >= 0.75 * peak and (j1 - i) * DT < hi_len / 2:
                j1 += 1
            a, b = min(j0 * DT, t - pre), max(j1 * DT, t + post)
        else:
            a, b = t - pre, t + post
        if b - a > hi_len:  # keep the peak inside, centred
            mid = (a + b) / 2 if ev is not None else t
            a, b = mid - hi_len / 2, mid + hi_len / 2
        if b - a < lo_len:
            grow = (lo_len - (b - a)) / 2
            a, b = a - grow, b + grow
        a, b = max(0.0, a), min(duration, b)
        ia = max(0, min(int(a / DT), s0) - 5)
        ib = min(n, max(int(b / DT), s1 + 1) + 5)
        score[ia:ib] = 0.0
        taken[ia:ib] = True
        sl = slice(int(a / DT), max(int(a / DT) + 1, int(b / DT)))
        sigs = {k: round(float(v[sl].max()), 3) for k, v in features.items()}
        picks.append(
            Pick(
                start=round(a, 2),
                end=round(b, 2),
                peak_t=t,
                peak=peak,
                kind=kind,
                signals=sigs,
                label=_label(kind, ev),
                event=ev,
            )
        )
    return _deoverlap(picks)


def _label(kind: str, ev: dict[str, Any] | None) -> str:
    if kind == "airtime" and ev:
        return f"jump {ev['air_s']:.1f}s"
    if kind == "crash" and ev:
        return f"crash {ev['peak_g']:.0f}g"
    return KIND_LABEL.get(kind, kind)


def _deoverlap(picks: list[Pick]) -> list[Pick]:
    """Trim overlaps in favour of the higher peak; drop clips that become too short."""
    keep: list[Pick] = []
    for p in sorted(picks, key=lambda p: -p.peak):
        for q in keep:
            if p.start < q.end and q.start < p.end:
                if p.peak_t < q.start:
                    p.end = min(p.end, q.start)
                else:
                    p.start = max(p.start, q.end)
        if p.end - p.start >= 0.6 * GEOMETRY.get(p.kind, GEOMETRY["vlm"])[0]:
            keep.append(p)
    return keep


def must_include_picks(
    src: Path, ctx: SkiContext, threshold: float, duration: float, offset: float = 0.0
) -> list[Pick]:
    """Clips for crashes at or above ``threshold`` confidence (``crashes.json``).

    Learned taste weights underweight rare events (a crash is a few seconds of
    a day), so a confident crash is added regardless of the weights: its
    planned clip (approach, impact, aftermath) and, when one was found, the
    reaction afterwards. ``threshold`` <= 0 turns this off.

    Args:
        src: Source path.
        ctx: Day context (``ctx.crashes``).
        threshold: Minimum crash confidence (0-1).
        duration: Length of this (part of the) source.
        offset: Start of a recovered part (results are part-relative).
    """
    if threshold <= 0:
        return []
    out: list[Pick] = []
    for c in ctx.crashes.get(_key(src), []):
        if c.get("confidence", 0.0) < threshold:
            continue
        spans = [("crash", c["clip"]["start"], c["clip"]["end"], c["t0"])]
        if c["clip"].get("reaction"):
            r = c["clip"]["reaction"]
            spans.append(("crash reaction", r["start"], r["end"], r["start"]))
        for label, a, b, peak_t in spans:
            a, b = max(0.0, a - offset), min(duration, b - offset)
            if b - a < 1.0:
                continue  # in another part of a recovered source
            out.append(
                Pick(
                    start=round(a, 2),
                    end=round(b, 2),
                    peak_t=peak_t - offset,
                    peak=float(c["confidence"]),
                    kind="wipeout",
                    signals={
                        "wipeout": round(float(c["confidence"]), 3),
                        **{k: round(float(v), 3) for k, v in c.get("scores", {}).items()},
                    },
                    label=label,
                    event={
                        "type": "crash",
                        "id": c.get("id"),
                        "t": c["t0"],
                        "end": c["t1"],
                        "confidence": c["confidence"],
                    },
                )
            )
    return out


def merge_must_include(picks: list[Pick], forced: list[Pick]) -> list[Pick]:
    """``forced`` plus the picks that do not overlap any of them."""
    keep = [p for p in picks if not any(p.start < f.end and f.start < p.end for f in forced)]
    return forced + keep


# --------------------------------------------------------------------------- persistence
def features_path(project: Path, src: Path, offset: float = 0.0) -> Path:
    """``<project>/ski_features/<stem>-<digest>.npz`` for one source (part)."""
    digest = hashlib.sha1(f"{_key(src)}|{offset:.3f}".encode()).hexdigest()[:8]
    return project / FEATURES_DIR / f"{source_path(src).stem}-{digest}.npz"


def save_features(
    project: Path,
    src: Path,
    offset: float,
    duration: float,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    friends: np.ndarray,
    events: list[dict[str, Any]],
) -> Path:
    """Store the grids so ``cutpoint refmatch`` can refit and re-rank without re-analysis."""
    path = features_path(project, src, offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(
        {"source": str(src), "offset": offset, "duration": duration, "events": events}
    )
    np.savez_compressed(
        path,
        mask=mask.astype(np.float32),
        friends=friends.astype(np.float32),
        meta=np.array(meta),
        **{f"f_{k}": v.astype(np.float32) for k, v in features.items()},
    )
    return path


def load_features(path: Path) -> dict[str, Any]:
    """Inverse of :func:`save_features`."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        return {
            **meta,
            "mask": z["mask"].astype(float),
            "friends": z["friends"].astype(float),
            "features": {k[2:]: z[k].astype(float) for k in z.files if k.startswith("f_")},
        }


def load_project_features(project: Path) -> list[dict[str, Any]]:
    """Every saved source of a ski project (offset-0 parts first)."""
    folder = project / FEATURES_DIR
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; run `cutpoint analyze --style ski` first")
    return sorted(
        (load_features(p) for p in folder.glob("*.npz")), key=lambda d: (d["source"], d["offset"])
    )
