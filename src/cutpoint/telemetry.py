"""GoPro GPMF telemetry (HERO 9-13): speed, airtime, crashes, lifts, on a 0.1 s grid.

Insta360 X3 clips (``.lrv`` / ``.insv`` sources) are read from the IMU trailer of
their ``_00_.insv`` (see :mod:`cutpoint.pano.x3`): accel and gyro only, no
GPS, so speed is NaN, stops come from low vibration and the start time comes
from the file name.

``telemetry-parser`` (gyroflow) reads the "GoPro MET" track of an MP4. It reads
the whole file, so it is fast from a local disk or the OS cache and slow from
slow or network storage: run ``cutpoint stage`` first (the parser is pointed at the
staged copy).

Per source, :func:`extract` returns a :class:`Telemetry` with these 0.1 s grids:

    speed_kmh   GPS9 3D speed, fix >= 3 and DOP < 5 only, 1 s median, short gaps
                interpolated; NaN where there is no usable fix
    alt_m       GPS altitude (MSL), smoothed; NaN without a fix
    accel_g     peak |accel| per bin (g); the sensor saturates near 8 g
    yaw_dps     gyro rate around the gravity axis (deg/s); gravity = 1 s low-pass
                of the accelerometer, so it is in the same frame as the gyro
    vibration   std of |accel| over 1 s (g): ~0 standing/riding a lift, high skiing
    carve       0-1: turn rhythm (std of yaw rate over 2 s) x speed factor
    airtime     0-1 plateau over each airtime event (take-off .. landing)
    crash       0-1 plateau over each crash event (impact followed by a stop)
    lift_mask   1 on a lift (GPS climb >= 0.5 m/s at < 30 km/h, held >= 30 s)
    still_mask  1 when stopped (GPS < 3 km/h, or no GPS and vibration < 0.04 g)
    wind        GoPro WindProcessing meter / 100
    faces       GoPro FACE detections: largest face box area (fraction of frame)
    indoor      SCEN "INDO" probability (lodge, gondola cabin)
    gps_ok      1 where speed_kmh is valid

Events (``Telemetry.events``) are ``{"type": "airtime"|"crash", "t", "end", ...}``
in source seconds: airtime has ``air_s``, ``height_m`` (g*T^2/8), ``landing_g``
and ``confidence`` (high >= 0.5 s, medium >= 0.3 s, low); crash has
``peak_g``, ``tilt_deg`` (gravity direction change: head pitch/roll), speeds
before/after and ``score``; an impact only counts when a stop follows.

:func:`join_runs` orders clips by their absolute start (GPS UTC, else the
camera's CDAT clock corrected by its TZON offset) and splits the day into runs
at lift rides and long gaps.
"""

from __future__ import annotations

import contextlib
import logging
import struct
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.config import source_path
from cutpoint.dsp import conv_same

logger = logging.getLogger(__name__)

DT = 0.1
G = 9.80665
TELEMETRY_VERSION = 2
_GPS_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class TelemetryParams:
    """Detection thresholds (all part of the cache key)."""

    min_fix: int = 3
    max_dop: float = 5.0
    max_speed_kmh: float = 140.0
    freefall_g: float = 0.45  # smoothed |a| below this is free fall
    min_air_s: float = 0.2
    landing_g: float = 2.5  # spike within 0.5 s after the free fall
    min_takeoff_kmh: float = 8.0  # with GPS: slower "free falls" are camera handling
    impact_g: float = 4.0
    crash_min_kmh: float = 12.0  # speed before an impact for it to count as a crash
    crash_score: float = 0.5
    lift_climb_mps: float = 0.5
    lift_max_kmh: float = 30.0
    lift_min_s: float = 30.0
    still_kmh: float = 3.0
    still_min_s: float = 4.0
    still_vibration_g: float = 0.04

    def key(self) -> str:
        """Short hash of the parameters + code version."""
        return cache.params_hash({**asdict(self), "v": TELEMETRY_VERSION})


@dataclass
class Telemetry:
    """Telemetry of one source on the analysis grid."""

    source: str
    duration: float
    grid: dict[str, np.ndarray]
    events: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Grid length."""
        return len(self.grid["speed_kmh"])

    def summary(self) -> dict[str, Any]:
        """Per-source numbers for logs and telemetry.json."""
        g = self.grid
        sp = g["speed_kmh"][g["gps_ok"] > 0]
        air = [e for e in self.events if e["type"] == "airtime"]
        return {
            "duration_s": round(self.duration, 1),
            "start_utc": self.meta.get("start_utc"),
            "start_local": self.meta.get("start_local"),
            "clock": self.meta.get("clock"),
            "gps_fix_share": round(float(g["gps_ok"].mean()), 3) if self.n else 0.0,
            "top_speed_kmh": round(float(sp.max()), 1) if sp.size else None,
            "p95_speed_kmh": round(float(np.percentile(sp, 95)), 1) if sp.size else None,
            "alt_range_m": (
                [round(float(np.nanmin(g["alt_m"])), 0), round(float(np.nanmax(g["alt_m"])), 0)]
                if np.isfinite(g["alt_m"]).any()
                else None
            ),
            "lift_s": round(float(g["lift_mask"].sum() * DT), 1),
            "still_s": round(float(g["still_mask"].sum() * DT), 1),
            "airtime_events": len(air),
            "airtime_high": sum(1 for e in air if e["confidence"] == "high"),
            "crash_events": sum(1 for e in self.events if e["type"] == "crash"),
            "max_g": round(float(g["accel_g"].max()), 2) if self.n else None,
            "faces_s": round(float((g["faces"] > 0).sum() * DT), 1),
            "imu_hz": self.meta.get("imu_hz"),
        }


# --------------------------------------------------------------------------- GPMF parsing
def _klv_blocks(raw: bytes) -> list[tuple[str, int, int, bytes]]:
    """Split a raw GPMF payload into ``(fourcc, size, repeat, body)`` blocks."""
    out, pos = [], 0
    while pos + 8 <= len(raw):
        fourcc = raw[pos : pos + 4].decode("latin1")
        size, repeat = raw[pos + 5], struct.unpack(">H", raw[pos + 6 : pos + 8])[0]
        n = size * repeat
        out.append((fourcc, size, repeat, raw[pos + 8 : pos + 8 + n]))
        pos += 8 + ((n + 3) & ~3)
    return out


def _payload_times(groups: list[dict[str, Any]], key: str) -> list[tuple[float, Any]]:
    """``(start_s, stream dict)`` for every payload carrying ``key``."""
    out = []
    for g in groups:
        v = g.get(key)
        if isinstance(v, dict) and "TimestampUs" in v:
            out.append((int(v["TimestampUs"]) / 1e6, v))
    return out


def _spread(times: list[float], counts: list[int], default_dt: float) -> list[np.ndarray]:
    """Sample times inside each payload: evenly between consecutive payload starts."""
    res = []
    for i, (t, n) in enumerate(zip(times, counts, strict=True)):
        span = (times[i + 1] - t) if i + 1 < len(times) else n * default_dt
        res.append(t + np.arange(n) * (span / max(n, 1)))
    return res


def _gps9(groups: list[dict[str, Any]]) -> np.ndarray:
    """GPS9 rows ``[t, lat, lon, alt, v2d, v3d, days, secs, dop, fix]`` (10 Hz)."""
    pays = _payload_times(groups, "GPS9")
    rows: list[list[float]] = []
    blocks = []
    for t, v in pays:
        raw = v.get("0x47505339")
        if raw is None:
            blocks.append((t, []))
            continue
        scale = v.get("Scale") or [1] * 9
        recs = []
        for fourcc, size, repeat, body in _klv_blocks(bytes(raw)):
            if fourcc != "GPS9" or size != 32:
                continue
            for i in range(repeat):
                r = struct.unpack(">lllllllHH", body[i * 32 : (i + 1) * 32])
                # Scale comes from the file: keep the old truncating zip on odd lengths.
                recs.append([a / b for a, b in zip(r, scale, strict=False)])
        blocks.append((t, recs))
    times = _spread([b[0] for b in blocks], [len(b[1]) for b in blocks], 0.1)
    for ts, (_, recs) in zip(times, blocks, strict=True):
        rows += [[float(t), *r] for t, r in zip(ts, recs, strict=True)]
    return np.asarray(rows, dtype=float).reshape(-1, 10)


def _series(
    groups: list[dict[str, Any]], key: str, default_dt: float
) -> tuple[np.ndarray, list[Any]]:
    """Times and ``Data`` items of a stream that telemetry-parser decodes itself."""
    pays = [(t, v.get("Data") or []) for t, v in _payload_times(groups, key)]
    times = _spread([p[0] for p in pays], [len(p[1]) for p in pays], default_dt)
    t_all = np.concatenate(times) if times else np.zeros(0)
    data = [d for _, items in pays for d in items]
    return t_all, data


def _scen_indoor(groups: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """SCEN classifier: times and the INDO (indoor) probability."""
    ts, vals = [], []
    for t, v in _payload_times(groups, "SCEN"):
        raw = v.get("0x5343454e")
        if raw is None:
            continue
        for fourcc, size, repeat, body in _klv_blocks(bytes(raw)):
            if fourcc != "SCEN" or size != 8:
                continue
            probs = {
                body[i * 8 : i * 8 + 4].decode("latin1"): struct.unpack(
                    ">f", body[i * 8 + 4 : i * 8 + 8]
                )[0]
                for i in range(repeat)
            }
            ts.append(t)
            vals.append(probs.get("INDO", 0.0))
    return np.asarray(ts), np.asarray(vals)


def _faces(groups: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """FACE detections: times and the largest box area (fraction of the frame)."""
    ts, area = [], []
    for t, v in _payload_times(groups, "FACE"):
        raw = v.get("0x46414345")
        if raw is None:
            continue
        best = 0.0
        for fourcc, size, repeat, body in _klv_blocks(bytes(raw)):
            if fourcc != "FACE" or size != 14:
                continue
            for i in range(repeat):  # BBSSSSSBB: ver, conf, id, x, y, w, h, ...
                _, _, _, _, _, w, h, _, _ = struct.unpack(">BBHHHHHBB", body[i * 14 : (i + 1) * 14])
                best = max(best, (w / 65535) * (h / 65535))
        ts.append(t)
        area.append(best)
    return np.asarray(ts), np.asarray(area)


def _start_clock(
    groups: list[dict[str, Any]], gps: np.ndarray
) -> tuple[datetime | None, int | None, str]:
    """Absolute start of the clip: GPS UTC, else camera CDAT minus its TZON offset."""
    default = groups[0].get("Default", {}) if groups else {}
    tz_min = None
    with contextlib.suppress(TypeError, ValueError):
        tz_min = int(default.get("0x545a4f4e"))
    if len(gps):
        ok = (gps[:, 9] >= 2) & (gps[:, 6] > 0)
        if ok.any():
            i = int(np.argmax(ok))
            utc = _GPS_EPOCH + timedelta(days=gps[i, 6], seconds=gps[i, 7])
            return utc - timedelta(seconds=gps[i, 0]), tz_min, "gps"
    try:
        cdat = int(default.get("0x43444154"))
    except (TypeError, ValueError):
        return None, tz_min, "none"
    local = datetime.fromtimestamp(cdat, tz=UTC)  # local wall time as epoch
    return local - timedelta(minutes=tz_min or 0), tz_min, "camera"


# --------------------------------------------------------------------------- grid helpers
def _bin_index(t: np.ndarray, n: int) -> np.ndarray:
    return np.clip((t / DT).astype(int), 0, n - 1)


def _bin_mean(t: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    """Mean of ``v`` per 0.1 s bin, NaN for empty bins."""
    idx = _bin_index(t, n)
    s = np.bincount(idx, weights=v, minlength=n)[:n]
    c = np.bincount(idx, minlength=n)[:n]
    out = np.full(n, np.nan)
    np.divide(s, c, out=out, where=c > 0)
    return out


def _bin_max(t: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf)
    np.maximum.at(out, _bin_index(t, n), v)
    out[~np.isfinite(out)] = np.nan
    return out


def _ffill_hold(x: np.ndarray, hold_bins: int) -> np.ndarray:
    """Forward-fill NaNs for at most ``hold_bins`` (sample-and-hold of slow streams)."""
    out = x.copy()
    last, age = np.nan, hold_bins + 1
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last, age = out[i], 0
        else:
            age += 1
            if age <= hold_bins:
                out[i] = last
    return out


def _interp_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN gaps no longer than ``max_gap`` bins."""
    out = x.copy()
    ok = np.isfinite(out)
    if ok.sum() < 2:
        return out
    idx = np.arange(len(out))
    filled = np.interp(idx, idx[ok], out[ok])
    i = 0
    while i < len(out):
        if not ok[i]:
            j = i
            while j < len(out) and not ok[j]:
                j += 1
            if i > 0 and j < len(out) and j - i <= max_gap:
                out[i:j] = filled[i:j]
            i = j
        else:
            i += 1
    return out


def _rolling(x: np.ndarray, k: int, fn: str = "mean") -> np.ndarray:
    """Centered rolling mean/std (NaN-aware) over ``k`` bins."""
    k = max(1, k)
    ok = np.isfinite(x).astype(float)
    xv = np.where(ok > 0, x, 0.0)
    ker = np.ones(k)
    c = conv_same(ok, ker)
    s = conv_same(xv, ker)
    mean = np.divide(s, c, out=np.full(len(x), np.nan), where=c > 0)
    if fn == "mean":
        return mean
    s2 = conv_same(xv * xv, ker)
    var = np.divide(s2, c, out=np.full(len(x), np.nan), where=c > 0) - mean**2
    return np.sqrt(np.clip(var, 0, None))


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k or k < 2:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    with np.errstate(all="ignore"):
        import warnings  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win, axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index pairs of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2], strict=True))  # padded mask: rises and falls pair up


def _clean_mask(mask: np.ndarray, close_bins: int, min_bins: int) -> np.ndarray:
    """Close gaps shorter than ``close_bins``, then drop runs shorter than ``min_bins``."""
    out = mask.astype(bool).copy()
    runs = _runs(out)
    for (_, e0), (s1, _) in pairwise(runs):
        if s1 - e0 <= close_bins:
            out[e0:s1] = True
    for s, e in _runs(out):
        if e - s < min_bins:
            out[s:e] = False
    return out


# --------------------------------------------------------------------------- detection
def _airtime_events(
    t: np.ndarray,
    am: np.ndarray,
    fs: float,
    speed: np.ndarray,
    gps_ok: np.ndarray,
    p: TelemetryParams,
) -> list[dict[str, Any]]:
    k = max(1, round(0.05 * fs))
    ams = conv_same(am, np.ones(k) / k)
    ff = ams < p.freefall_g
    runs = _runs(ff)
    merged: list[list[int]] = []
    for s, e in runs:  # bridge 60 ms flickers
        if merged and (t[s] - t[merged[-1][1] - 1]) < 0.06:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    n = len(speed)
    for s, e in merged:
        air = float(t[e - 1] - t[s])
        if air < p.min_air_s:
            continue
        j = min(len(am), e + int(0.5 * fs))
        landing = float(am[e:j].max()) if j > e else 0.0
        i = min(n - 1, int(t[s] / DT))
        v = float(speed[i]) if gps_ok[i] > 0 and np.isfinite(speed[i]) else None
        if v is not None and v < p.min_takeoff_kmh:
            continue  # camera dropped / handled while standing
        landed = landing >= p.landing_g
        conf = "high" if air >= 0.5 else "medium" if air >= 0.3 else "low"
        if not landed and conf != "high":
            conf = "low"
        out.append(
            {
                "type": "airtime",
                "t": round(float(t[s]), 2),
                "end": round(float(t[e - 1]), 2),
                "air_s": round(air, 2),
                "height_m": round(G * air * air / 8, 2),
                "landing_g": round(landing, 2),
                "landed": landed,
                "speed_kmh": round(v, 1) if v is not None else None,
                "confidence": conf,
            }
        )
    return out


def _crash_events(
    t: np.ndarray,
    am: np.ndarray,
    grav: np.ndarray,
    gyro_mag: np.ndarray,
    speed: np.ndarray,
    vib: np.ndarray,
    airtime: list[dict[str, Any]],
    p: TelemetryParams,
) -> list[dict[str, Any]]:
    """Impacts (>= ``impact_g``) that end in a stop.

    A high-g spike alone is common while skiing (bumps, pole plants, hard
    landings), and the camera orientation changes whenever a helmet-mounted
    rider looks around, so neither counts on its own. A crash needs a stop
    right after the impact: GPS speed from >= ``crash_min_kmh`` to <= 5 km/h
    (or 25%), or, without GPS, stillness (vibration < ``still_vibration_g``).
    ``tilt_deg`` is the change of the gravity direction in the camera frame
    (head pitch/roll: lying in the snow) and raises the score; looking
    around is yaw and leaves gravity nearly unchanged.
    """
    peaks = np.flatnonzero(am >= p.impact_g)
    if not peaks.size:
        return []
    groups: list[list[int]] = [[int(peaks[0])]]
    for i in peaks[1:]:
        if t[i] - t[groups[-1][-1]] <= 1.0:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    n = len(speed)
    out = []
    for grp in groups:
        i_pk = max(grp, key=lambda i: am[i])
        tp = float(t[i_pk])

        def med(lo: float, hi: float, arr: np.ndarray = speed) -> float | None:
            a0, a1 = max(0, int(lo / DT)), min(n, int(hi / DT))
            seg = arr[a0:a1]
            seg = seg[np.isfinite(seg)]
            return float(np.median(seg)) if seg.size else None

        v_before, v_after = med(tp - 3, tp - 0.5), med(tp + 1.5, tp + 5.0)
        if v_before is not None and v_after is not None:
            moving = v_before >= p.crash_min_kmh
            stopped = moving and v_after <= max(p.still_kmh + 2, 0.25 * v_before)
        else:  # no GPS: moving before (vibration), still after
            moving = (med(tp - 3, tp - 0.5, vib) or 0) > 0.12
            stopped = moving and (med(tp + 1.5, tp + 4.0, vib) or 1) < p.still_vibration_g
        if not stopped:
            continue
        pre = (t >= tp - 2.5) & (t <= tp - 0.5)
        post = (t >= tp + 1.5) & (t <= tp + 4.0)
        tilt = 0.0
        if pre.any() and post.any():
            g0, g1 = grav[pre].mean(0), grav[post].mean(0)
            cosang = float(np.dot(g0, g1) / (np.linalg.norm(g0) * np.linalg.norm(g1) + 1e-9))
            tilt = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        w = (t >= tp - 0.5) & (t <= tp + 1.5)
        spin = float(gyro_mag[w].max()) if w.any() else 0.0
        landing_of = next((a for a in airtime if 0 <= tp - a["end"] <= 0.7), None)
        score = (
            0.3 * float(np.clip((am[i_pk] - 3.5) / 4.5, 0, 1))
            + 0.4
            + 0.3 * float(np.clip(tilt / 70, 0, 1))
        )
        if score < p.crash_score:
            continue
        out.append(
            {
                "type": "crash",
                "t": round(tp - 0.5, 2),
                "end": round(tp + 2.5, 2),
                "impact_t": round(tp, 2),
                "peak_g": round(float(am[i_pk]), 2),
                "tilt_deg": round(tilt, 0),
                "stopped": True,
                "spin_dps": round(spin, 0),
                "speed_before_kmh": round(v_before, 1) if v_before is not None else None,
                "speed_after_kmh": round(v_after, 1) if v_after is not None else None,
                "after_jump": landing_of is not None,
                "score": round(score, 2),
            }
        )
    return out


def _plateau(
    n: int, events: list[dict[str, Any]], value_key: str | None, pad_s: float = 0.3
) -> np.ndarray:
    x = np.zeros(n)
    for e in events:
        a, b = int(max(0.0, e["t"] - pad_s) / DT), int((e["end"] + pad_s) / DT) + 1
        v = 1.0 if value_key is None else float(e[value_key])
        x[a : min(n, b)] = np.maximum(x[a : min(n, b)], v)
    return x


# --------------------------------------------------------------------------- main entry
def _compute(path: Path, duration: float, p: TelemetryParams) -> Telemetry | None:
    import telemetry_parser as tp_mod  # noqa: PLC0415

    t0 = time.perf_counter()
    try:
        parser = tp_mod.Parser(str(path))
        groups = parser.telemetry()
        imu = parser.normalized_imu()
    except Exception as err:  # noqa: BLE001 - the Rust parser raises plain Exceptions
        logger.info("%s: no readable GoPro telemetry (%s)", path.name, err)
        return None
    # A sample can lack one sensor (seen on HERO13: 1 of 79k with accl None).
    imu = [s for s in imu or [] if s.get("accl") is not None and s.get("gyro") is not None]
    if not imu or not groups:
        logger.info("%s: no IMU telemetry", path.name)
        return None
    n = max(1, int(duration / DT) + 1)
    t = np.array([s["timestamp_ms"] for s in imu]) / 1000.0
    acc = np.array([s["accl"] for s in imu], dtype=float)
    gyr = np.array([s["gyro"] for s in imu], dtype=float)
    fs = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    am = np.linalg.norm(acc, axis=1) / G

    # Gravity-aligned yaw rate: project the gyro onto the 1 s low-passed accel.
    k = max(1, int(fs))
    grav = np.stack([conv_same(acc[:, i], np.ones(k) / k) for i in range(3)], 1)
    grav /= np.linalg.norm(grav, axis=1, keepdims=True) + 1e-9
    yaw = (gyr * grav).sum(1)
    gyro_mag = np.linalg.norm(gyr, axis=1)

    grid: dict[str, np.ndarray] = {}
    grid["accel_g"] = np.nan_to_num(_bin_max(t, am, n), nan=1.0)
    grid["yaw_dps"] = np.nan_to_num(_bin_mean(t, yaw, n))
    am_bin = _bin_mean(t, am, n)
    grid["vibration"] = np.nan_to_num(_rolling(am_bin, int(1.0 / DT), "std"))

    gps = _gps9(groups)
    speed = np.full(n, np.nan)
    alt = np.full(n, np.nan)
    if len(gps):
        ok = (
            (gps[:, 9] >= p.min_fix)
            & (gps[:, 8] < p.max_dop)
            & (gps[:, 5] * 3.6 <= p.max_speed_kmh)
        )
        g_ok = gps[ok]
        if len(g_ok):
            speed = _bin_mean(g_ok[:, 0], g_ok[:, 5] * 3.6, n)
            alt = _bin_mean(g_ok[:, 0], g_ok[:, 3], n)
            speed = _interp_gaps(_ffill_hold(speed, 1), 30)
            alt = _interp_gaps(_ffill_hold(alt, 1), 30)
            speed = _rolling_median(speed, 10)
            alt = _rolling_median(alt, 30)
    gps_ok = np.isfinite(speed).astype(float)
    grid["speed_kmh"], grid["alt_m"], grid["gps_ok"] = speed, alt, gps_ok

    # Carve: turn rhythm (yaw-rate std over 2 s) scaled by speed.
    yaw_std = np.nan_to_num(_rolling(grid["yaw_dps"], int(2.0 / DT), "std"))
    spd0 = np.nan_to_num(speed)
    speed_fac = (
        np.clip((spd0 - 10) / 20, 0, 1)
        if gps_ok.any()
        else np.clip((grid["vibration"] - 0.1) / 0.2, 0, 1)
    )
    grid["carve"] = np.clip(yaw_std / 45.0, 0, 1) * speed_fac

    # Lift: sustained GPS climb at modest speed; still: stopped.
    lift = np.zeros(n, bool)
    if np.isfinite(alt).any():
        h = int(15 / DT)
        a_pad = np.pad(alt, h, mode="edge")
        climb = (a_pad[2 * h :] - a_pad[: -2 * h]) / 30.0
        lift = (np.nan_to_num(climb) >= p.lift_climb_mps) & (spd0 <= p.lift_max_kmh)
        lift = _clean_mask(lift, int(20 / DT), int(p.lift_min_s / DT))
    still_gps = (gps_ok > 0) & (spd0 < p.still_kmh)
    still_imu = (gps_ok == 0) & (grid["vibration"] < p.still_vibration_g)
    still = _clean_mask(still_gps | still_imu, int(1.0 / DT), int(p.still_min_s / DT))
    still &= ~lift
    grid["lift_mask"], grid["still_mask"] = lift.astype(float), still.astype(float)

    # Slow streams (sample-and-hold up to their own period).
    wt, wd = _series(groups, "WindProcessing", 0.1)
    grid["wind"] = np.zeros(n)
    if len(wt):
        meter = np.array([d[1] if isinstance(d, list | tuple) else 0 for d in wd], float)
        grid["wind"] = np.nan_to_num(_ffill_hold(_bin_mean(wt, meter / 100, n), 2))
    ft, fa = _faces(groups)
    grid["faces"] = np.nan_to_num(_ffill_hold(_bin_max(ft, fa, n), 9)) if len(ft) else np.zeros(n)
    st, sv = _scen_indoor(groups)
    grid["indoor"] = (
        np.nan_to_num(_ffill_hold(_bin_mean(st, sv, n), 10)) if len(st) else np.zeros(n)
    )

    airtime = _airtime_events(t, am, fs, speed, gps_ok, p)
    airtime = [e for e in airtime if not lift[min(n - 1, int(e["t"] / DT))]]
    crashes = _crash_events(t, am, grav, gyro_mag, speed, grid["vibration"], airtime, p)
    w_air = {"high": 1.0, "medium": 0.7, "low": 0.35}
    for e in airtime:
        e["strength"] = round(w_air[e["confidence"]] * float(np.clip(e["air_s"] / 0.8, 0.5, 1)), 2)
    grid["airtime"] = _plateau(n, airtime, "strength")
    grid["crash"] = _plateau(n, crashes, "score")

    start, tz_min, clock = _start_clock(groups, gps)
    meta = {
        "camera": f"{parser.camera} {parser.model}".strip(),
        "imu_hz": round(fs, 1),
        "gps_samples": len(gps),
        "start_utc": start.isoformat() if start else None,
        "start_local": (
            (start + timedelta(minutes=tz_min)).replace(tzinfo=None).isoformat(timespec="seconds")
            if start and tz_min is not None
            else None
        ),
        "tz_offset_min": tz_min,
        "clock": clock,
        "parse_s": round(time.perf_counter() - t0, 2),
    }
    events = sorted(airtime + crashes, key=lambda e: e["t"])
    return Telemetry(source=str(path), duration=duration, grid=grid, events=events, meta=meta)


def _x3_clock(tel: Telemetry, insv00: Path) -> None:
    """Start time of an Insta360 clip from its file name (camera clock, local time).

    The X3 trailer has no GPS and no GoPro CDAT/TZON, so the name
    (``VID_YYYYMMDD_HHMMSS_..``) is the only clock; it is interpreted in this
    computer's time zone, which only matters for ordering and run joining.
    """
    from cutpoint.pano.x3 import parse_name  # noqa: PLC0415

    f = parse_name(insv00)
    if f is None or tel.meta.get("clock") not in (None, "none"):
        return
    local = datetime.strptime(f["date"] + f["time"], "%Y%m%d%H%M%S")
    tel.meta.update(
        {
            "start_local": local.isoformat(timespec="seconds"),
            "start_utc": local.astimezone(UTC).isoformat(),
            "clock": "filename",
            "imu_file": insv00.name,
        }
    )


def cached_speed(src: Path, params: TelemetryParams | None = None) -> np.ndarray | None:
    """GPS speed (km/h, NaN where invalid) on the 0.1 s grid, from the cache only.

    Never parses the file (that reads all of it), so it is safe to call for
    any source; None when no telemetry has been extracted yet or there is none.
    """
    p = params or TelemetryParams()
    npz = cache.source_dir(src) / f"telemetry-{p.key()}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as z:
        if "speed_kmh" not in z.files:
            return None
        speed = z["speed_kmh"].astype(np.float64)
        if "gps_ok" in z.files:
            speed = np.where(z["gps_ok"] > 0, speed, np.nan)
    return speed


def extract(
    src: Path, duration: float, params: TelemetryParams | None = None, force: bool = False
) -> Telemetry | None:
    """Telemetry of ``src`` (cached per source), or None if it has none.

    Reads the staged copy when ``cutpoint stage`` has one; otherwise warns, because
    the parser reads the entire file.

    Args:
        src: Source video (original path; ``source_remap`` and the stage apply).
        duration: Container duration (s); sets the grid length.
        params: Detection thresholds.
        force: Ignore the cache.
    """
    from cutpoint.config import remap_source  # noqa: PLC0415
    from cutpoint.stage import is_remote, staged_path, storage_label  # noqa: PLC0415

    p = params or TelemetryParams()
    cdir = cache.source_dir(src)
    stem = f"telemetry-{p.key()}"
    npz, meta_path = cdir / f"{stem}.npz", cdir / f"{stem}.json"
    if not force and npz.exists() and meta_path.exists():
        data = cache.load_json(meta_path)
        if data is not None:
            if data.get("none"):
                return None
            with np.load(npz) as z:
                grid = {k: z[k] for k in z.files}
            return Telemetry(
                source=str(src),
                duration=data["duration"],
                grid=grid,
                events=data["events"],
                meta=data["meta"],
            )
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    io = remap_source(src)
    x3_imu = imu_source(io)  # Insta360 X3: IMU trailer of the _00_.insv (cheap to read)
    read_path = x3_imu or Path(staged_path(str(io)))
    if x3_imu is None and read_path == io and is_remote(io):
        logger.warning(
            "%s is not staged: telemetry-parser reads the whole file from %s "
            "(run `cutpoint stage` first for speed)",
            src.name,
            storage_label(io),
        )
    tel = _compute(read_path, duration, p)
    if tel is not None and x3_imu is not None:
        _x3_clock(tel, x3_imu)
    if tel is None:
        cache.save_json(meta_path, {"none": True})
        np.savez_compressed(npz)
        return None
    tel.source = str(src)
    np.savez_compressed(npz, **{k: v.astype(np.float32) for k, v in tel.grid.items()})
    cache.save_json(meta_path, {"duration": duration, "events": tel.events, "meta": tel.meta})
    logger.info(
        "%s: telemetry %s (%.1fs): top %.0f km/h, %d airtime, %d crash, lift %.0fs",
        src.name,
        tel.meta.get("camera"),
        tel.meta["parse_s"],
        tel.summary()["top_speed_kmh"] or 0,
        tel.summary()["airtime_events"],
        tel.summary()["crash_events"],
        tel.summary()["lift_s"],
    )
    return tel


# --------------------------------------------------------------------------- runs / day
def join_runs(tels: list[Telemetry], max_gap_s: float = 240.0) -> dict[str, Any]:
    """Order clips by absolute start and split the day into runs.

    A run is a stretch of skiing (not lift, not long stops) across clips. It
    ends at a lift segment inside a clip or at an unfilmed gap longer than
    ``max_gap_s`` (typically a lift ride with the camera off).

    Returns:
        ``{"clips": [...], "runs": [...], "day": {...}}``.
    """
    clips = []
    for tel in tels:
        start = tel.meta.get("start_utc")
        st = datetime.fromisoformat(start).timestamp() if start else None
        clips.append(
            {
                "source": tel.source,
                "start_epoch": st,
                "duration": tel.duration,
                "start_local": tel.meta.get("start_local"),
                "clock": tel.meta.get("clock"),
            }
        )
    order = sorted(
        range(len(tels)),
        key=lambda i: (clips[i]["start_epoch"] is None, clips[i]["start_epoch"] or 0, i),
    )
    segs: list[dict[str, Any]] = []  # skiing segments in absolute time
    for i in order:
        tel, c = tels[i], clips[i]
        g = tel.grid
        moving = (g["lift_mask"] == 0) & (g["still_mask"] == 0)
        for s, e in _runs(_clean_mask(moving, int(30 / DT), int(5 / DT))):
            base = c["start_epoch"]
            segs.append(
                {
                    "source": tel.source,
                    "t0": s * DT,
                    "t1": e * DT,
                    "a0": base + s * DT if base is not None else None,
                    "a1": base + e * DT if base is not None else None,
                    "lift_after": bool(e < tel.n and g["lift_mask"][e : e + int(60 / DT)].any()),
                }
            )
    runs: list[list[dict[str, Any]]] = []
    for s in segs:
        prev = runs[-1][-1] if runs else None
        if (
            prev is not None
            and not prev["lift_after"]
            and s["a0"] is not None
            and prev["a1"] is not None
            and s["a0"] - prev["a1"] <= max_gap_s
        ):
            runs[-1].append(s)
        else:
            runs.append([s])
    by_src = {tel.source: tel for tel in tels}
    out_runs = []
    for k, run in enumerate(runs, 1):
        sp, alt, dur = [], [], 0.0
        for s in run:
            g = by_src[s["source"]].grid
            a, b = int(s["t0"] / DT), int(s["t1"] / DT)
            v = g["speed_kmh"][a:b]
            sp.append(v[np.isfinite(v)])
            al = g["alt_m"][a:b]
            alt.append(al[np.isfinite(al)])
            dur += s["t1"] - s["t0"]
        sp_all = np.concatenate(sp) if sp else np.zeros(0)
        alt_all = np.concatenate(alt) if alt else np.zeros(0)
        out_runs.append(
            {
                "run": k,
                "segments": [
                    {"source": s["source"], "start": round(s["t0"], 1), "end": round(s["t1"], 1)}
                    for s in run
                ],
                "clips": list(dict.fromkeys(source_path(s["source"]).name for s in run)),
                "start_local": next(
                    (c["start_local"] for c in clips if c["source"] == run[0]["source"]), None
                ),
                "skiing_s": round(dur, 1),
                "top_speed_kmh": round(float(sp_all.max()), 1) if sp_all.size else None,
                "vertical_m": round(float(alt_all.max() - alt_all.min()), 0)
                if alt_all.size
                else None,
            }
        )
    sums = [t.summary() for t in tels]
    tops = [s["top_speed_kmh"] for s in sums if s["top_speed_kmh"] is not None]
    day = {
        "clips": len(tels),
        "footage_s": round(sum(t.duration for t in tels), 1),
        "top_speed_kmh": max(tops) if tops else None,
        "runs": len(out_runs),
        "airtime_events": sum(s["airtime_events"] for s in sums),
        "airtime_high": sum(s["airtime_high"] for s in sums),
        "crash_events": sum(s["crash_events"] for s in sums),
        "lift_s": round(sum(s["lift_s"] for s in sums), 1),
        "still_s": round(sum(s["still_s"] for s in sums), 1),
    }
    return {"clips": [clips[i] for i in order], "runs": out_runs, "day": day}
