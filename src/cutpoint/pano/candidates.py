"""Director paths of the analysis candidates: cache, diagnostics and read API (#517).

At the end of ``cutpoint 360 analyze``, :func:`cache_candidate_paths` plans the director
path of every 360 candidate in ``highlights.json`` (16:9, the settings ``clips`` uses: the
moment-kind tweaks, then ``pano.director`` of config.json) from the cached view layer, so
the UI can preview a moment through the virtual camera and flag poor paths before export.
No video is decoded: the view layer, the voice layer and the IMU orientation are read from
their caches.

Cache: one JSON per candidate in ``<project>/pano_candidates/`` (:data:`CACHE_DIR`), named
after the candidate's slot (clip, source, start, end, 16:9 / 9:16;
:func:`slot_name`), holding::

    {"version": 2, "key": {...}, "candidate": {"id", "source", "start", "end", "kind",
     "label"}, "diagnostics": {...}, "path": {... the director path ...}}

It is a folder of its own, not ``pano_paths/<name>/``: those names are the user's (clips
folders, ``360 director --out-name``) and their path files may be hand-edited for
``extend``, so a cache there could collide with them and could not be pruned safely.
Cutpoint owns this folder: each analysis removes the files of candidates it no longer has.

``key`` (:func:`path_key`) is everything the path depends on: the source, the exact span,
16:9 / 9:16, the kind, the effective :class:`~cutpoint.pano.director.DirectorParams`,
:data:`~cutpoint.pano.director.PATH_VERSION`, :data:`CACHE_VERSION`, the Cutpoint
version, the code identity (:func:`code_id`: a hash of the source of the modules that plan
and score the path, so a code change without a version bump plans anew), the clip's
duration / fps / frames and the identity (name, size, modification time) of the cached view
layer, voice layer and orientation track it reads.
``clips`` (:class:`~cutpoint.pano.render.PanoExporter`) reuses a cached path only when the
key it would plan with is equal (:func:`reusable_path`); any difference (a span moved by
tails, padding or the speech pass, other settings, a recomputed view layer) plans it anew,
as before. A reused path is the very dict the director returned, so the render is the same.

Diagnostics (:func:`diagnostics`; shares of the segment, sampled at :data:`DIAG_FPS`)::

    seconds          segment length
    quality          score / pass / fails of the path's geometric QA (qa_paths.path_quality)
    above_horizon    view centre above the horizon (pitch > ABOVE_HORIZON_DEG)
    sky              horizon below the frame's lower third line (SKY_HORIZON_LINE):
                     more than two thirds of the picture is above the horizon
    peak_yaw_rate    fastest yaw change within a shot (deg/s, at RATE_FPS; cuts excluded)
    yaw_travel       total absolute yaw change within shots (deg, the same samples as
                     peak_yaw_rate; cuts excluded)
    yaw_travel_per_min   yaw_travel per minute (deg/min)
    net_turns        largest net rotation in one direction within any SPIN_WINDOW_S window
                     (the whole segment if shorter), in full turns: the range (max - min) of
                     the yaw accumulated within shots (cuts add nothing) over the window
    cuts, cuts_per_min   hard cuts inside the segment (flagged from CUTS_MIN cuts on)
    wearer_in_view   the wearer's axis (path wearer_axis) inside the frame; samples whose
                     nearest key is a selfie shot count as not in view
    off_target       steps whose view misses a yaw bin that scores clearly better
                     (OFF_TARGET_MARGIN) than the one it looks at; selfie steps excluded
    flags            stable codes, see FLAGS

Angles are the director's display frame (horizon-levelled from the IMU, see
:mod:`cutpoint.pano.director`). Flag thresholds are **provisional** (set from first
principles; to be tuned on a project with known-bad director cuts); :func:`path_flags`
applies the current ones, so the read API reflects a change without re-planning. The flag
codes are a stable API: :data:`FLAGS`.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, state
from cutpoint.pano import director as D
from cutpoint.pano import geometry as G

logger = logging.getLogger(__name__)

CACHE_DIR = "pano_candidates"
CACHE_VERSION = 2  # bump when the file layout, the key or the diagnostics change
# (2: yaw_travel_per_min, net_turns)
DIAG_FPS = 10.0  # path samples per second for the shares
RATE_FPS = 30.0  # ... for the yaw rate (as qa.pan_metrics)

# Provisional thresholds (to be tuned on footage with known-bad director cuts).
ABOVE_HORIZON_DEG = 0.0  # view centre pitch above this: above the horizon
SKY_HORIZON_LINE = 1 / 3  # horizon this share of the half height below the centre (the
# frame's lower third line) or lower: more than two thirds of the picture is sky
SKY_MAX_SHARE = 0.2  # flag ``sky`` above this share of the segment
WHIP_DEG_S = 55.0  # flag ``whip`` above this peak yaw rate (the QA's max_pan_max)
SPIN_WINDOW_S = 30.0  # ``net_turns``: rolling window (s)
SPIN_TURNS = 1.0  # flag ``spin`` from this many net turns in one window (a full 360 in 30 s) ...
SPIN_TRAVEL_DEG_MIN = 540.0  # ... or above this yaw travel per minute (9 deg/s on average) ...
SPIN_MIN_TRAVEL_DEG = 360.0  # ... with at least a full turn of travel in the segment (a
# few seconds of one deliberate pan is over the rate, but no spinning camera) ...
SPIN_TRAVEL_MIN_TURNS = 0.5  # ... and at least this many net turns (half a turn one way in a
# window): gentle sweeps back and forth travel far but go nowhere
CUTS_MAX_PER_MIN = 6.0  # flag ``cuts`` above this many hard cuts per minute ...
CUTS_MIN = 2  # ... and from this many cuts on (one cut in a short span is no cut-heavy path)
WEARER_MAX_SHARE = 0.25  # flag ``wearer`` above this share with the wearer in frame
OFF_TARGET_MARGIN = 0.25  # a bin scoring this much more (about the full forward bias,
# w_forward) than the one looked at, outside the frame, is a missed target
OFF_TARGET_MAX_SHARE = 0.4  # flag ``off_target`` above this share of the steps

# Stable flag codes (the UI shows a "camera path looks poor" badge with the reason).
FLAGS = {
    "sky": "looks at the sky",
    "whip": "whip pans",
    "spin": "keeps turning",
    "cuts": "many hard cuts",
    "wearer": "shows the wearer",
    "off_target": "misses the best view",
    "qa_fail": "fails the geometric direction QA",
}


PREVIEW_VERSION = 1  # preview_key: bump when previews render differently
# The pano settings (config.json) a preview's pixels depend on besides its path (preview_key).
PREVIEW_SETTINGS = ("imu_offset_s", "level_strength", "tilt_follow_s", "max_follow_deg")

# Modules whose code plans or scores a director path (compute_path, its scores, voice and
# orientation inputs and the QA it records); their source is hashed into the key.
CODE_MODULES = (
    "cutpoint.pano.director",
    "cutpoint.pano.geometry",
    "cutpoint.pano.views",
    "cutpoint.pano.qa",
    "cutpoint.pano.qa_paths",
    "cutpoint.pano.x3",
    "cutpoint.events.voice",
    "cutpoint.events.visual",
    "cutpoint.events.common",
)


# --------------------------------------------------------------------------- key
@state.register_cache
@functools.cache
def code_id() -> str:
    """Identity of the path-planning code: sha1 of the source of :data:`CODE_MODULES` (12 hex
    characters; computed once per process). ``__version__`` alone stays the same across
    installs from git, so a code change would otherwise reuse paths planned by older code."""
    h = hashlib.sha1()
    for name in CODE_MODULES:
        spec = importlib.util.find_spec(name)
        h.update(name.encode("utf-8"))
        if spec is not None and spec.origin:
            with contextlib.suppress(OSError):
                h.update(Path(spec.origin).read_bytes())
    return h.hexdigest()[:12]


def _file_id(path: Path) -> list[Any] | None:
    """``[name, size, mtime_ns]`` of a cache file, or None when it does not exist."""
    try:
        st = path.stat()
    except OSError:
        return None
    return [path.name, st.st_size, st.st_mtime_ns]


def _inputs(src: Path) -> dict[str, Any]:
    """Identity of the cached layers a path of ``src`` reads.

    Raises:
        OSError: If the source is not reachable (its cache folder needs its size and mtime).
    """
    from cutpoint.events import voice as VO  # noqa: PLC0415
    from cutpoint.pano import views as VW  # noqa: PLC0415

    views = VW.cache_path(src)
    return {
        "dir": views.parent.name,  # the source's cache folder: path, size and mtime
        "views": _file_id(views),
        "voice": _file_id(VO.cache_path(src, VO.VoiceParams())),
        "orient": _file_id(G._orient_path(src)),
    }


def effective_params(kind: str | None, overrides: dict[str, Any] | None) -> D.DirectorParams:
    """The settings :func:`director.compute_path` plans ``kind`` with (no base params)."""
    p = D.params_for(kind)
    return D.with_params(p, overrides) if overrides else p


def path_key(
    clip: Any,
    start: float,
    end: float,
    kind: str | None,
    overrides: dict[str, Any] | None,
    vertical: bool = False,
) -> dict[str, Any]:
    """Everything the director path of ``[start, end]`` depends on (JSON-normalised).

    Raises:
        OSError: If the source is not reachable.
        ValueError: On bad ``overrides``.
    """
    from cutpoint import __version__  # noqa: PLC0415

    key = {
        "v": CACHE_VERSION,
        "path_version": D.PATH_VERSION,
        "cutpoint": __version__,
        "code": code_id(),
        "source": str(clip.source),
        "clip": [clip.clip_id, clip.duration, clip.fps, clip.frames],
        "start": float(start),
        "end": float(end),
        "vertical": bool(vertical),
        "kind": kind,
        "params": asdict(effective_params(kind, overrides)),
        "inputs": _inputs(Path(clip.source)),
    }
    return json.loads(json.dumps(key))


def slot_name(clip: Any, start: float, end: float, vertical: bool = False) -> str:
    """File name of a candidate's cache entry (one per clip, span and aspect)."""
    slot = json.dumps([str(clip.source), float(start), float(end), bool(vertical)])
    return f"{clip.clip_id}_{hashlib.sha1(slot.encode('utf-8')).hexdigest()[:12]}.json"


def _entry(folder: Path, name: str) -> dict[str, Any] | None:
    return _valid(cache.load_json(folder / name))


def _valid(entry: Any) -> dict[str, Any] | None:
    """``entry`` (a loaded cache file) if it is a current, well-formed entry, else None."""
    if not isinstance(entry, dict) or entry.get("version") != CACHE_VERSION:
        return None
    if not isinstance(entry.get("path"), dict) or not isinstance(entry.get("key"), dict):
        return None
    return entry


#: Reasons :func:`cached_path` gives for a key mismatch, in this order: key fields to reason.
_MISS_REASONS = (
    (("kind", "params"), "settings changed"),
    (("clip", "inputs"), "layers changed"),
    (("v", "path_version", "cutpoint", "code"), "planning code changed"),
    (("source", "start", "end"), "span differs from 360 analyze"),
)


def _miss_reason(cached: dict[str, Any], key: dict[str, Any]) -> str:
    """Why the cached key ``cached`` differs from ``key`` (reasons joined by ``, ``)."""
    out = [r for fields, r in _MISS_REASONS if any(cached.get(f) != key.get(f) for f in fields)]
    return ", ".join(out) or "key changed"


def cached_path(
    project: Path,
    clip: Any,
    start: float,
    end: float,
    kind: str | None,
    overrides: dict[str, Any] | None,
    vertical: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """:func:`reusable_path` with the reason a cached path is not reused.

    Returns ``(path, None)`` on a hit, ``(None, None)`` when ``360 analyze`` cached no
    paths in ``project`` and ``(None, reason)`` otherwise: ``span differs from 360 analyze``
    (no entry for this exact span, such as one moved by a tail or padding), ``9:16 not
    planned by 360 analyze`` (``vertical``: ``360 analyze`` plans 16:9 only), ``settings
    changed``, ``layers changed``, ``planning code changed`` (see :data:`_MISS_REASONS`),
    ``older cache`` or ``source not reachable``. Reads no more than :func:`reusable_path`.
    """
    folder = project / CACHE_DIR
    if not folder.is_dir():
        return None, None
    data = cache.load_json(folder / slot_name(clip, start, end, vertical))
    if data is None:  # no entry for this span (or an unreadable one)
        reason = "9:16 not planned by 360 analyze" if vertical else "span differs from 360 analyze"
        return None, reason
    entry = _valid(data)
    if entry is None:
        return None, "older cache"
    try:
        key = path_key(clip, start, end, kind, overrides, vertical)
    except OSError:
        return None, "source not reachable"
    except ValueError:
        return None, "settings changed"
    if entry["key"] == key:
        return entry["path"], None
    return None, _miss_reason(entry["key"], key)


def plan_path(
    project: Path,
    clip: Any,
    start: float,
    end: float,
    kind: str | None,
    overrides: dict[str, Any] | None,
    vertical: bool = False,
) -> tuple[dict[str, Any], bool, str | None]:
    """The director path ``clips`` renders ``[start, end]`` with when no path file is reused:
    the one ``360 analyze`` cached when :func:`cached_path` reuses it, else a new
    :func:`director.compute_path` (orientation and view layer loaded as there). Writes nothing.

    Returns:
        ``(path, cached, reason)``: ``reason`` is :func:`cached_path`'s (None on a hit, or
        when ``360 analyze`` cached no paths).

    Raises:
        Exception: Whatever :func:`director.compute_path` raises (footage offline, no IMU).
    """
    path, why = cached_path(project, clip, start, end, kind, overrides, vertical)
    if path is not None:
        return path, True, None
    path = D.compute_path(clip, start, end, vertical=vertical, kind=kind, overrides=overrides)
    return path, False, why


def reusable_path(
    project: Path,
    clip: Any,
    start: float,
    end: float,
    kind: str | None,
    overrides: dict[str, Any] | None,
    vertical: bool = False,
) -> dict[str, Any] | None:
    """The cached path of ``[start, end]`` when it was planned with exactly what
    :func:`director.compute_path` would use now (:func:`path_key`), else None."""
    return cached_path(project, clip, start, end, kind, overrides, vertical)[0]


# --------------------------------------------------------------------------- diagnostics
def _share(x: np.ndarray) -> float | None:
    return round(float(np.mean(x)), 3) if len(x) else None


def _sky(pitch: np.ndarray, fov: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Per sample: the horizon at or below the frame's lower third line (pitch + = up)."""
    half_v = np.arctan(np.tan(np.radians(fov) / 2) * size[1] / size[0])
    p = np.radians(np.clip(pitch, -89.9, 89.9))
    return np.tan(p) > np.tan(half_v) * SKY_HORIZON_LINE


def _in_frame(
    d: np.ndarray, yaw: np.ndarray, pitch: np.ndarray, fov: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Per sample: display ray ``d[i]`` inside the view ``(yaw, pitch, fov)[i]``."""
    out = np.zeros(len(d), bool)
    th = np.tan(np.radians(fov) / 2)
    tv = th * size[1] / size[0]
    for i in range(len(d)):
        o = G.view_matrix(float(yaw[i]), float(pitch[i])).T @ d[i]  # x right, y down, z fwd
        if o[2] > 1e-6:
            out[i] = abs(o[0] / o[2]) <= th[i] and abs(o[1] / o[2]) <= tv[i]
    return out


def _selfie(path: dict[str, Any], times: np.ndarray) -> np.ndarray:
    """Per sample: the nearest key of the path is a (deliberate) selfie shot."""
    keys = path["keys"]
    selfie = np.array([k.get("target") == "selfie" for k in keys], bool)
    if not selfie.any():
        return np.zeros(len(times), bool)
    kt = np.array([k["t"] for k in keys], float)
    return selfie[np.abs(times[:, None] - kt[None, :]).argmin(1)]


def _off_target(
    path: dict[str, Any], grid: dict[str, Any], start: float, end: float
) -> float | None:
    """Share of the segment's steps whose view misses a clearly better yaw bin."""
    times, cen, ys = grid.get("times"), grid.get("centres"), grid.get("yaw_score")
    if times is None or cen is None or ys is None or not len(times):
        return None
    keys = path["keys"]
    hits = []
    for i, t in enumerate(times):
        if not start - 1e-6 <= t <= end + 1e-6 or i >= len(keys):
            continue
        k = keys[i]
        if k.get("target") == "selfie":
            continue
        best = int(np.argmax(ys[i]))
        mine = int(np.argmin(np.abs(G.wrap_deg(cen - k["yaw"]))))
        out = abs(float(G.wrap_deg(cen[best] - k["yaw"]))) > k["fov"] / 2
        hits.append(out and ys[i, best] - ys[i, mine] > OFF_TARGET_MARGIN)
    return _share(np.asarray(hits, bool))


def diagnostics(
    path: dict[str, Any],
    start: float,
    end: float,
    *,
    grid: dict[str, Any] | None = None,
    orient: G.Orientation | None = None,
) -> dict[str, Any]:
    """Compact quality record of a path's ``[start, end]`` (see the module docstring).

    ``grid``: the director's score grid (``compute_path(grid=)``; without it
    ``off_target`` is None); ``orient``: the clip's orientation track (without it
    ``wearer_in_view`` is None).
    """
    dur = max(end - start, 0.0)
    size = (1080, 1920) if path.get("vertical") else (1920, 1080)
    times = start + np.arange(0.0, dur, 1.0 / DIAG_FPS) if dur > 0 else np.array([start])
    yaw, pitch, fov = D.evaluate(path, times)
    wearer = None
    axis = path.get("wearer_axis")
    if orient is not None and axis is not None:
        a = np.asarray(axis, float)
        d = np.stack([orient.r_vw(float(t)).T @ a for t in times])
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        wearer = _share(_in_frame(d, yaw, pitch, fov, size) & ~_selfie(path, times))
    rt = start + np.arange(0.0, max(dur, 1.0 / RATE_FPS), 1.0 / RATE_FPS)
    ry, _, _ = D.evaluate(path, rt)
    seg = D.shot_index(path, rt)
    step = np.where(seg[1:] == seg[:-1], np.diff(ry), 0.0)  # yaw change within shots
    rate = np.abs(step[seg[1:] == seg[:-1]]) * RATE_FPS
    travel = float(np.abs(step).sum())
    cuts = [c for c in path.get("cuts", []) if start < c < end]
    q = path.get("quality")
    diag = {
        "seconds": round(dur, 3),
        "quality": {k: q.get(k) for k in ("score", "pass", "fails")} if q else None,
        "above_horizon": _share(pitch > ABOVE_HORIZON_DEG),
        "sky": _share(_sky(pitch, fov, size)),
        "peak_yaw_rate": round(float(rate.max()), 1) if len(rate) else 0.0,
        "yaw_travel": round(travel, 1),
        "yaw_travel_per_min": round(travel / max(dur / 60.0, 1e-6), 1),
        "net_turns": _net_turns(step),
        "cuts": len(cuts),
        "cuts_per_min": round(len(cuts) / max(dur / 60.0, 1e-6), 2),
        "wearer_in_view": wearer,
        "off_target": _off_target(path, grid, start, end) if grid else None,
    }
    diag["flags"] = path_flags(diag)
    return diag


def _net_turns(step: np.ndarray) -> float:
    """Largest net rotation (turns) within any :data:`SPIN_WINDOW_S` window of the yaw
    steps ``step`` (at :data:`RATE_FPS`): the range of their running sum per window."""
    acc = np.concatenate([[0.0], np.cumsum(step)])
    win = min(len(acc), round(SPIN_WINDOW_S * RATE_FPS) + 1)
    w = np.lib.stride_tricks.sliding_window_view(acc, win)
    return round(float((w.max(1) - w.min(1)).max()) / 360.0, 2)


def path_flags(diag: dict[str, Any]) -> list[str]:
    """:data:`FLAGS` codes a diagnostics record exceeds (current thresholds; a missing
    measure never flags)."""

    def num(name: str) -> float | None:
        v = diag.get(name)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def over(name: str, limit: float) -> bool:
        v = num(name)
        return v is not None and v > limit

    out = []
    if over("sky", SKY_MAX_SHARE):
        out.append("sky")
    if over("peak_yaw_rate", WHIP_DEG_S):
        out.append("whip")
    turns = num("net_turns")
    if (turns is not None and turns >= SPIN_TURNS) or (
        over("yaw_travel_per_min", SPIN_TRAVEL_DEG_MIN)
        and (num("yaw_travel") or 0.0) >= SPIN_MIN_TRAVEL_DEG
        and turns is not None
        and turns >= SPIN_TRAVEL_MIN_TURNS
    ):
        out.append("spin")
    if over("cuts_per_min", CUTS_MAX_PER_MIN) and over("cuts", CUTS_MIN - 1):
        out.append("cuts")
    if over("wearer_in_view", WEARER_MAX_SHARE):
        out.append("wearer")
    if over("off_target", OFF_TARGET_MAX_SHARE):
        out.append("off_target")
    if (diag.get("quality") or {}).get("pass") is False:
        out.append("qa_fail")
    return out


# --------------------------------------------------------------------------- 360 analyze
def _candidates(project: Path) -> list[dict[str, Any]]:
    hl = cache.load_json(project / "highlights.json")
    cands = hl.get("candidates") if isinstance(hl, dict) else None
    return [c for c in cands or [] if isinstance(c, dict)]


def cache_candidate_paths(project: Path) -> dict[str, Any]:
    """Plan and cache the 16:9 director path of every 360 candidate of ``highlights.json``.

    A candidate whose cached entry has the current key is kept as is. Candidates of clips
    without a view layer (footage offline too) are skipped, keeping their entry; a candidate
    whose path fails (or whose clip's orientation cannot be loaded) logs a WARNING and is
    skipped. Never raises for a path; files of candidates no longer listed are removed (and
    the folder, once empty).

    Returns:
        Stats: ``candidates`` (360 ones), ``computed``, ``cached``, ``skipped``,
        ``failed``, ``seconds``, ``flags`` (code -> candidates) and ``dir`` (or None).
    """
    from cutpoint.pano.project import clip_for, load  # noqa: PLC0415
    from cutpoint.pano.views import load_views  # noqa: PLC0415
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    t0 = time.perf_counter()
    folder = project / CACHE_DIR
    stats: dict[str, Any] = {
        "candidates": 0,
        "computed": 0,
        "cached": 0,
        "skipped": 0,
        "failed": 0,
        "flags": {},
        "dir": None,
        "seconds": 0.0,
    }
    clips = load(project)
    todo = [(c, clip) for c in _candidates(project) if (clip := clip_for(clips, c["source"]))]
    stats["candidates"] = len(todo)
    try:
        overrides = D.config_overrides() or None
    except ValueError as err:
        logger.warning("Director paths not planned: %s", err)
        stats["skipped"] = len(todo)
        return stats
    layers: dict[str, tuple[Any, Any] | None] = {}
    keep: set[str] = set()
    flags: Counter[str] = Counter()
    for c, clip in todo:
        # as clips passes them (the path records them rounded, so 4 and 4.0 would differ)
        cid, start, end, kind = c.get("id"), c.get("start"), c.get("end"), c.get("kind")
        try:
            name = slot_name(clip, start, end)
            keep.add(name)  # an entry kept while its clip is offline is still checked by clips
            if clip.source not in layers:
                layers[clip.source] = None  # a failed load is reported once per clip
                src = Path(clip.source)
                pv = load_views(src)
                if pv is None:
                    logger.info("clip %s: no 360 view layer; paths not planned", clip.clip_id)
                else:
                    orient = G.orientation(src, imu_source(src), clip.fps, max(clip.frames, 1))
                    layers[clip.source] = (pv, orient)
            if layers[clip.source] is None:
                stats["skipped"] += 1
                continue
            pv, orient = layers[clip.source]
            key = path_key(clip, start, end, kind, overrides)
            entry = _entry(folder, name) if folder.is_dir() else None
            if entry is not None and entry["key"] == key:
                stats["cached"] += 1
            else:
                grid: dict[str, Any] = {}
                path = D.compute_path(
                    clip,
                    start,
                    end,
                    kind=kind,
                    overrides=overrides,
                    orient=orient,
                    pv=pv,
                    grid=grid,
                )
                entry = {
                    "version": CACHE_VERSION,
                    "key": key,
                    "candidate": {
                        "id": cid,
                        "source": c["source"],
                        "start": start,
                        "end": end,
                        "kind": kind,
                        "label": c.get("label"),
                    },
                    "diagnostics": diagnostics(path, start, end, grid=grid, orient=orient),
                    "path": path,
                }
                folder.mkdir(exist_ok=True)
                cache.save_json(folder / name, entry, indent=1)
                stats["computed"] += 1
            flags.update(path_flags(entry["diagnostics"]))
        except Exception as err:  # noqa: BLE001 - a path never fails the analysis
            stats["failed"] += 1
            logger.warning("%s: director path not planned (%s: %s)", cid, type(err).__name__, err)
    _prune(folder, keep)
    stats["flags"] = dict(sorted(flags.items()))
    stats["dir"] = folder if folder.is_dir() else None
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    if todo:
        logger.info(
            "Director paths: %d candidate(s), %d planned, %d cached, %d skipped, %d failed "
            "in %.1fs%s",
            len(todo),
            stats["computed"],
            stats["cached"],
            stats["skipped"],
            stats["failed"],
            stats["seconds"],
            "; flagged: " + ", ".join(f"{k} {v}" for k, v in stats["flags"].items())
            if flags
            else "",
        )
    return stats


def _prune(folder: Path, keep: set[str]) -> None:
    """Remove the cache files not in ``keep`` (and ``folder`` once empty)."""
    if not folder.is_dir():
        return
    for f in folder.glob("*.json"):
        if f.name not in keep:
            try:
                f.unlink()
            except OSError as err:
                logger.warning("could not remove stale %s: %s", f, err)
    with contextlib.suppress(OSError):  # only when empty
        folder.rmdir()


# --------------------------------------------------------------------------- read API
def _current(entry: dict[str, Any], clip: Any, overrides: Any) -> bool | None:
    """Whether ``entry`` was planned with today's inputs (None: cannot be checked)."""
    cand = entry.get("candidate") or {}
    try:
        key = path_key(
            clip,
            entry["key"]["start"],
            entry["key"]["end"],
            cand.get("kind"),
            overrides,
            bool(entry["key"].get("vertical")),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return key == entry["key"]


def candidate_path(project: Path, cid: str) -> dict[str, Any] | None:
    """The cached director path of candidate ``cid`` of ``highlights.json`` (16:9).

    The candidate is looked up by id in the current ``highlights.json`` and its entry by
    its source and span, so a re-ranked analysis (new ids) still finds it.

    Returns:
        None when the candidate, its 360 clip or its cache entry is missing; else
        ``id``, ``file`` (the cache file), ``path`` (render it with
        :func:`cutpoint.pano.render.render_segment`), ``diagnostics`` (``flags`` by the
        current thresholds, with ``reasons``: code -> text) and ``current`` (True: planned
        with today's settings and layers, what ``clips`` would reuse; False: stale, ``clips``
        plans it anew; None: cannot be checked, e.g. footage offline).
    """
    from cutpoint.pano.project import clip_for, load  # noqa: PLC0415

    cand = next((c for c in _candidates(project) if c.get("id") == cid), None)
    if cand is None:
        return None
    clip = clip_for(load(project), cand["source"])
    if clip is None:
        return None
    folder = project / CACHE_DIR
    file = folder / slot_name(clip, float(cand["start"]), float(cand["end"]))
    entry = _entry(folder, file.name)
    if entry is None:
        return None
    try:
        overrides = D.config_overrides() or None
    except ValueError:
        overrides = None
        current = None
    else:
        current = _current(entry, clip, overrides)
    diag = dict(entry.get("diagnostics") or {})
    diag["flags"] = path_flags(diag)
    diag["reasons"] = {f: FLAGS[f] for f in diag["flags"]}
    return {"id": cid, "file": file, "path": entry["path"], "diagnostics": diag, "current": current}


def path_id(path: dict[str, Any]) -> str:
    """Short stable identity of a path's content: 12 hex characters of the sha1 of its
    canonical JSON (sorted keys). Equal paths give equal ids; any change gives another."""
    text = json.dumps(path, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def preview_key(path: dict[str, Any], *, vertical: bool, fmt: Any, settings: dict[str, Any]) -> str:
    """Stable identity of a preview render, to name its file: 16 hex characters of the sha1
    of the canonical JSON (sorted keys) of everything that changes its pixels. Equal inputs
    give equal keys on every run and OS.

    It covers:

    - :func:`path_id` of ``path`` (the views, its ``level``, ``vertical`` and span);
    - ``vertical`` (9:16 or 16:9);
    - ``fmt`` (a :class:`~cutpoint.pano.render.Format360`): ``width``, ``height``, ``fps``;
    - of ``settings`` (the ``pano`` section of config.json,
      :func:`~cutpoint.pano.project.pano_config`), :data:`PREVIEW_SETTINGS`: ``imu_offset_s``
      (the IMU orientation track, :func:`~cutpoint.pano.render.clip_orientation`) and
      ``level_strength``, ``tilt_follow_s``, ``max_follow_deg`` (the horizon levelling,
      :func:`~cutpoint.pano.render.stabilised_r_vw`), as given (an absent key and its
      default value give different keys; the render is the same);
    - :data:`PREVIEW_VERSION`.

    Not covered, as a preview does not depend on them: any other setting, such as
    ``pano.director`` (it changes the path, so :func:`path_id`), ``pano.cmd_every`` (previews
    render with ``cmd_every=1``) and ``pano.equirect_yaw_offset`` (it turns levelled, stitched
    sources only; previews render from the unlevelled ``.lrv``).
    """
    blob = {
        "version": PREVIEW_VERSION,
        "path": path_id(path),
        "vertical": bool(vertical),
        "format": [int(fmt.width), int(fmt.height), int(fmt.fps)],
        "settings": {k: settings.get(k) for k in PREVIEW_SETTINGS},
    }
    text = json.dumps(blob, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def candidate_flags(project: Path) -> dict[str, dict[str, Any]]:
    """Per candidate id with a cache entry, for badges on every moment at once (no freshness
    check: see :func:`candidate_path`):

    ``file`` (the cache file), ``flags`` (:data:`FLAGS` codes by the current thresholds),
    ``reasons`` (code -> text), ``diagnostics`` (the stored record, its ``flags`` replaced by
    the current ones) and ``key`` (:func:`path_id` of the cached path: it changes whenever
    the cached path does, so it can name a rendered preview).

    Reads only ``pano.json``, ``highlights.json`` and the cache files (on POSIX the case
    check of a source's folder, :func:`cutpoint.config.folds_case`, may stat that folder once
    per process)."""
    from cutpoint.pano.project import clip_for, load  # noqa: PLC0415

    folder = project / CACHE_DIR
    if not folder.is_dir():
        return {}
    clips = load(project)
    out: dict[str, dict[str, Any]] = {}
    for c in _candidates(project):
        clip = clip_for(clips, c["source"])
        if clip is None:
            continue
        name = slot_name(clip, float(c["start"]), float(c["end"]))
        entry = _entry(folder, name)
        if entry is not None:
            diag = dict(entry.get("diagnostics") or {})
            diag["flags"] = flags = path_flags(diag)
            out[str(c.get("id"))] = {
                "file": folder / name,
                "flags": flags,
                "reasons": {f: FLAGS[f] for f in flags},
                "diagnostics": diag,
                "key": path_id(entry["path"]),
            }
    return out


# --------------------------------------------------------------------------- preview API
_LOCKS: dict[str, threading.Lock] = {}  # per clip source: one preview_path at a time
_LOCKS_GUARD = threading.Lock()


def _clip_lock(source: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(source, threading.Lock())


def _kind_for(project: Path, clip: Any, start: float, end: float) -> str | None:
    """The ``kind`` of the ``highlights.json`` candidate of ``clip`` overlapping ``[start,
    end]`` most (the moment being previewed, possibly trimmed), else None."""
    from cutpoint.pano.project import clip_for, load  # noqa: PLC0415

    clips = load(project)
    best, kind = 0.0, None
    for c in _candidates(project):
        try:
            ov = min(end, float(c["end"])) - max(start, float(c["start"]))
        except (KeyError, TypeError, ValueError):
            continue
        hit = clip_for(clips, c.get("source", ""))
        if ov > best and hit is not None and hit.clip_id == clip.clip_id:
            best, kind = ov, c.get("kind")
    return kind


def _level_forward(
    clip: Any,
    start: float,
    end: float,
    kind: str | None,
    overrides: dict[str, Any] | None,
    vertical: bool,
    why: str,
) -> dict[str, Any]:
    """A constant forward view: yaw at the travel direction (:func:`views.forward_yaw`,
    0 without a view layer), the director's default pitch and wide FOV, levelled by the IMU
    as any render (``level`` ``imu``) or, without an orientation track, for the nominal
    mount (``level`` :data:`~cutpoint.pano.render.UPRIGHT`). Planned over the span with
    :data:`~cutpoint.pano.director.PATH_MARGIN_S` on both sides, as a director path
    (:func:`~cutpoint.pano.director.planned_span`)."""
    from cutpoint.pano import render as R  # noqa: PLC0415
    from cutpoint.pano.views import forward_yaw, load_views  # noqa: PLC0415

    try:
        p = effective_params(kind, overrides)
    except ValueError:
        p = D.params_for(kind)
    level, yaw = "imu", 0.0
    try:
        R.clip_orientation(clip)  # the track render_segment levels with (now cached)
    except Exception as err:  # noqa: BLE001 - no IMU: the upright horizon
        # footage offline: levelled by the IMU once it is back (render_segment reads it)
        if _reachable(clip) and (not isinstance(err, OSError) or _no_imu(clip)):
            level = R.UPRIGHT
            why += f"; no IMU orientation ({type(err).__name__}), levelled for the upright mount"
    else:
        with contextlib.suppress(Exception):  # no view layer: yaw 0 (the camera's forward)
            pv = load_views(Path(clip.source))
            yaw = forward_yaw(pv) if pv is not None else 0.0
    fov = p.vertical_fov_wide if vertical else p.fov_wide
    t0, t1 = D.planned_span(clip, start, end)
    keys = [
        {"t": round(t, 3), "yaw": round(yaw, 2), "pitch": p.pitch_default, "fov": fov}
        | {"target": "forward"}
        for t in (t0, t1)
    ]
    path = {
        "version": D.PATH_VERSION,
        "clip": clip.clip_id,
        "source": str(clip.source),
        "segment": {"start": round(start, 3), "end": round(end, 3)},
        "planned": {"start": round(t0, 3), "end": round(t1, 3)},
        "vertical": vertical,
        "kind": kind,
        "forward_yaw": round(yaw, 1),
        "level": level,
        "params": asdict(p),
        "cuts": [],
        "seconds_by_target": {"forward": round(t1 - t0, 1)},
        "keys": keys,
    }
    return {"path": path, "kind": "level_forward", "reason": why}


def _reachable(clip: Any) -> bool:
    """The clip's source can be stat-ed (through ``source_remap``)."""
    from cutpoint.config import remap_source  # noqa: PLC0415

    try:
        remap_source(clip.source).stat()
    except OSError:
        return False
    return True


def _no_imu(clip: Any) -> bool:
    """``clip`` has no ``_00_.insv`` to read the IMU from (False when that cannot be checked;
    meaningful only for a :func:`_reachable` clip: an offline folder has no files either)."""
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    try:
        return imu_source(clip.source) is None
    except OSError:  # the folder cannot be checked: assume the IMU is there
        return False


def _err_path(err: OSError) -> Path | None:
    """The path an OSError names (its ``filename``), or None when it names none (or a file
    descriptor, such as ``os.stat(9)``)."""
    try:
        return Path(os.fsdecode(err.filename)) if err.filename is not None else None
    except (TypeError, ValueError):
        return None


def _is_source(clip: Any, err: OSError) -> bool:
    """``err`` names the clip's source, as given or through ``source_remap``."""
    from cutpoint.config import remap_source  # noqa: PLC0415

    got = _err_path(err)
    if got is None:
        return False
    try:
        return got in (Path(clip.source), remap_source(clip.source))
    except (OSError, TypeError, ValueError):
        return False


def _failure(clip: Any, err: Exception) -> str:
    """Why the director could not plan ``clip`` (``err``), for ``preview_path``'s reason:
    ``footage offline`` only when the source itself cannot be stat-ed or opened; another
    missing file (a model, a layer) is ``missing file: <name>``, any other OSError its class
    and message."""
    if not _reachable(clip):
        what = err.strerror if isinstance(err, OSError) else None
        return f"footage offline ({what or 'source not reachable'})"
    if _no_imu(clip):
        return "no IMU (no _00_.insv next to the clip)"
    if isinstance(err, OSError):
        if _is_source(clip, err):
            return f"footage offline ({err.strerror or type(err).__name__})"
        got = _err_path(err)
        name = got.name if got is not None else None
        if isinstance(err, FileNotFoundError) and name:
            return f"missing file: {name}"
        what = err.strerror or str(err) or "no message"
        if name:
            what += f", {name}"
        return f"director could not plan ({type(err).__name__}: {what})"
    return f"director could not plan ({type(err).__name__}: {err})"


def preview_path(
    project: Path,
    source: str | Path,
    start: float,
    end: float,
    *,
    vertical: bool = False,
) -> dict[str, Any]:
    """The virtual-camera path to preview ``[start, end]`` of a 360 clip with (16:9, or 9:16
    with ``vertical``); render it with :func:`cutpoint.pano.render.render_segment`.

    Returns:
        ``{"path": ..., "kind": ..., "reason": ...}``, ``kind`` one of:

        - ``cached``: the path ``360 analyze`` planned, reused exactly as ``clips`` reuses it
          (:func:`cached_path`: same span, settings, layers and code); ``reason`` None.
        - ``planned``: planned now by the director as ``clips`` would (:func:`plan_path`:
          ``pano.director`` of config.json, the moment's kind tweaks, the cached orientation
          and view layer); ``reason`` says why the cached one is not used
          (:func:`cached_path`'s text, or ``not planned by 360 analyze``). It is not written
          to the analysis cache: that holds the candidates of ``highlights.json`` only, and
          ``360 analyze`` prunes anything else.
        - ``level_forward``: the director cannot plan (footage offline, no IMU, bad
          ``pano.director`` settings, any other error): a constant forward view (travel
          direction, the director's default pitch and wide FOV), levelled by the IMU like any
          render or, without an orientation track, for the nominal upright mount (path
          ``level`` ``upright``); never the raw fisheye. ``reason`` says why.

        The kind (``laughter``, ``people``, ...) is that of the ``highlights.json`` candidate
        of the clip overlapping the span most.

    Never raises for footage or layer problems.

    Raises:
        ValueError: If ``source`` is not a 360 clip of ``project`` or ``end <= start``.

    Thread safety: safe from worker threads. Calls on the same clip run one at a time (a
    per-clip lock), so the caches they fill (orientation track, view scores) are written
    once; calls on different clips run in parallel. It changes no working directory and no
    logging setup. The per-process memo caches it fills (view scores, wearer axis, the CLIP
    text model of the view scores, config.json) are shared; two clips at once may compute
    the same entry twice.

    Source access (all kinds): besides ``pano.json``, ``highlights.json`` and the cache files,
    it stats the source (``clip.source``, through ``source_remap``) several times: the
    cache folders of its layers are keyed by its size and mtime (:func:`cache.source_dir`),
    and so is the freshness of a cached path. On POSIX the source's folder may also be
    stat-ed once per process (:func:`cutpoint.config.folds_case`). When the orientation
    track is not cached yet, ``planned`` and ``level_forward`` look up the clip's
    ``_00_.insv`` (which may list the source's folder) and read its IMU trailer. An
    unmounted source fails fast (``level_forward``, ``footage offline``); a hung network
    mount blocks the calling thread in that stat.
    """
    from cutpoint.pano.project import clip_for, load  # noqa: PLC0415

    start, end = float(start), float(end)
    if not end > start:
        raise ValueError(f"preview span {start:g}-{end:g} s is empty")
    clip = clip_for(load(project), source)
    if clip is None:
        raise ValueError(f"{source} is not a 360 clip of {project}")
    kind = _kind_for(project, clip, start, end)
    with _clip_lock(str(clip.source)):
        try:
            overrides = D.config_overrides() or None
        except ValueError as err:
            return _level_forward(clip, start, end, kind, None, vertical, f"settings: {err}")
        try:
            path, cached, why = plan_path(project, clip, start, end, kind, overrides, vertical)
        except Exception as err:  # noqa: BLE001 - a preview never fails on the footage
            return _level_forward(clip, start, end, kind, overrides, vertical, _failure(clip, err))
    if cached:
        return {"path": path, "kind": "cached", "reason": None}
    return {"path": path, "kind": "planned", "reason": why or "not planned by 360 analyze"}
