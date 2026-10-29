"""Coordinate frames, ffmpeg ``v360`` conventions and the IMU orientation track.

Frames (all right-handed, x right, y down, z forward, as in ``v360``):

``V`` (camera)
    ``v360``'s own frame for ``input=dfisheye``: the *right* half of a
    dual-fisheye picture is the lens looking along +z, the left half the lens
    looking along -z (mirrored in x). For the X3 that means +z = the ``_10_``
    lens (down at the skier on the user's mount) and -z = ``_00_`` (sky).
    Measured with a coordinate-encoded test image (ffmpeg 9.0.2).
``W`` (display world)
    Horizon-levelled: -y is up (from the fused IMU), yaw 0 is the camera's
    forward direction (``FORWARD_V``, where the skis point), smoothed over
    ``heading_smooth_s`` so the view does not swing with every turn. A clip
    filmed upright (lenses horizontal: handheld, on a pole) has ``FORWARD_V``
    near vertical; its yaw 0 is the ``_10_`` lens axis ``LENS_V`` instead
    (:func:`forward_axis`, one choice per clip).

``v360`` rotation (measured): with ``rorder=ypr``, an output ray ``d`` samples
the input along ``Ry(yaw) @ Rp(pitch) @ Rr(roll) @ d``, where ``Ry`` turns +z
toward +x, ``Rp`` turns +z toward -y (up) and ``Rr`` turns +x toward +y. We
render with ``rorder=ryp`` (``Rr @ Ry @ Rp``) because for a level camera the
middle angle stays near 0, which keeps the Euler decomposition away from
gimbal lock (the vertical lens axis puts ``pitch`` near 90 deg).

IMU: telemetry-parser's ``normalized_imu`` (accel m/s^2, gyro deg/s) maps to
``V`` by ``M_IMU_TO_V`` (V = (-z, -y, -x) of the IMU), verified by levelling
frames at 6 tilted moments of clip 035; the gyro sign was verified by
propagating gravity through quiet rotations (2.7 deg median error vs 22 deg
with the opposite sign).
"""

from __future__ import annotations

import io
import logging
import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache

logger = logging.getLogger(__name__)

ORIENT_VERSION = 2
LEVEL_VERSION = 1  # the up / heading part of the track (unchanged since v1; views cache key)
M_IMU_TO_V = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
FORWARD_V = np.array([0.0, -1.0, 0.0])  # where the skis point (camera "front")
LENS_V = np.array([0.0, 0.0, 1.0])  # the _10_ lens: yaw 0 of an upright clip (forward_axis)
FORWARD_MIN_HORIZONTAL = 0.5  # forward_axis keeps FORWARD_V while its median |horizontal| >= this
# The wearer seen from the downward lens: helmet near the +y edge of the
# _10_ circle, body toward its centre. Measured from person detections (the
# 366 boxes taller than 60 deg on clip 030 have median direction
# (0.09, 0.63, 0.69)); detections within a cone around it are the wearer.
WEARER_V = np.array([0.05, 0.68, 0.73]) / np.linalg.norm([0.05, 0.68, 0.73])
WEARER_CONE_DEG = 50.0
DFISHEYE_FOV = 200.0  # v360 ih_fov / iv_fov used for analysis and rendering
G = 9.80665


# --------------------------------------------------------------------------- rotations
def wrap_deg(d: np.ndarray | float) -> np.ndarray:
    """Angle(s) in degrees wrapped to ``[-180, 180)``."""
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def rot_y(deg: float) -> np.ndarray:
    """``v360`` yaw: turns +z toward +x."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_p(deg: float) -> np.ndarray:
    """``v360`` pitch: turns +z toward -y (up)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_r(deg: float) -> np.ndarray:
    """``v360`` roll: turns +x toward +y."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


SDK_EQUIRECT_FROM_CAMERA = rot_r(-0.85) @ rot_p(2.0)
"""Camera ``V`` -> SDK equirect (stitched without FlowState): ``e = C @ v``.

The renderer composes it with the per-frame IMU levelling for such stitches
(:func:`cutpoint.pano.render.command_lines`). Measured, not specified: read
by ``v360 input=equirect``, these stitches line up with our ``V`` frame turned
by about 2 deg, at every frame. Calibrated on clip 031 of the trip-360 project (SDK 3.1.5, optical flow +
stitchfusion, 5760x2880): at 32 times (0.5-16 s, every 0.5 s) the stitch was
registered against the ``_00_`` | ``_10_`` dual fisheye resampled through
``C^T`` (gradient-magnitude NCC 0.68-0.80, full 3D rotation search). The
per-time estimates stay within 0.84 deg (RMS 0.42 deg) of the joint fit
``Rr(-0.85) @ Rp(2.0)`` while the camera tilt ranged 4-37 deg and its heading
70 deg, so the rotation is constant and the IMU levelling composes as for
``dfisheye``. The residual likely absorbs lens calibration (about 1 deg per
lens in the clip metadata) and our 200 deg equidistant fisheye model.
"""


def decompose_ryp(r: np.ndarray) -> tuple[float, float, float]:
    """``(yaw, pitch, roll)`` in degrees with ``r == Rr(roll) @ Ry(yaw) @ Rp(pitch)``.

    This is ``v360``'s ``rorder=ryp``. Singular only when ``|yaw|`` = 90.
    """
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def view_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Rotation of a virtual camera looking at display ``yaw`` / ``pitch`` (deg)."""
    return rot_y(yaw) @ rot_p(pitch)


def level_matrix(up_v: np.ndarray, fwd_v: np.ndarray = FORWARD_V) -> np.ndarray:
    """``R_VW``: display-world vectors -> camera vectors for a camera whose up is ``up_v``.

    World -y is ``up_v``; world +z is ``fwd_v`` projected onto the horizontal.
    """
    u = up_v / (np.linalg.norm(up_v) + 1e-12)
    f = fwd_v - (fwd_v @ u) * u
    nf = np.linalg.norm(f)
    if nf < 1e-6:  # forward axis vertical: pick any horizontal direction
        f = np.cross(u, [1.0, 0.0, 0.0])
        nf = np.linalg.norm(f)
    f = f / nf
    y = -u
    x = np.cross(y, f)
    return np.stack([x, y, f], 1)


def axis_name(v: np.ndarray) -> str:
    """``"-y"`` for :data:`FORWARD_V`, ``"+z"`` for :data:`LENS_V` (meta and cache keys)."""
    return "+z" if np.array_equal(v, LENS_V) else "-y"


def forward_axis(up: np.ndarray) -> np.ndarray:
    """The camera axis whose horizontal direction is a clip's yaw 0, from its ``up`` track.

    :data:`FORWARD_V` (the helmet mount's front) while its median horizontal
    part is at least :data:`FORWARD_MIN_HORIZONTAL`; else :data:`LENS_V` when
    that lies more level (an upright camera, whose ``FORWARD_V`` points near
    up or down, so its horizontal part is tiny and its direction noise). One
    axis per clip, so the yaw 0 never jumps; whichever of the two is more
    level is at least 45 deg from vertical. Its sign: on upright raft footage the
    ``_10_`` lens (+z) faced the travel direction (the ``_00_`` lens the paddlers).
    """
    up = np.asarray(up, float).reshape(-1, 3)
    if not len(up):
        return FORWARD_V
    hy = float(np.median(np.sqrt(np.clip(1.0 - (up @ FORWARD_V) ** 2, 0.0, 1.0))))
    if hy >= FORWARD_MIN_HORIZONTAL:
        return FORWARD_V
    hz = float(np.median(np.sqrt(np.clip(1.0 - (up @ LENS_V) ** 2, 0.0, 1.0))))
    return LENS_V if hz > hy else FORWARD_V


def flat_fov(h_fov: float, width: int, height: int) -> float:
    """Vertical FOV (deg) of a rectilinear view with horizontal FOV ``h_fov``."""
    return math.degrees(2 * math.atan(math.tan(math.radians(h_fov) / 2) * height / width))


# --------------------------------------------------------------------------- orientation track
@dataclass
class Orientation:
    """Per-video-frame camera orientation of one clip.

    Attributes:
        t: Frame times (s), ``(N,)``.
        up: Up direction in camera (``V``) coordinates, ``(N, 3)``.
        heading: Integrated camera heading about the vertical (deg, unwrapped;
            arbitrary zero; increases when the camera turns left).
        heading_smooth: Low-passed ``heading`` (the display's yaw 0).
        meta: Fusion info.
        q: Full camera -> world rotation per frame (unit quaternions ``(w, x,
            y, z)``, ``(N, 4)``; world -y is up, its yaw zero arbitrary), read
            ``meta["q_offset_s"]`` after each frame time; used by the renderer.
            ``up`` / ``heading`` stay the director's frame.
        fwd: The camera axis of yaw 0 (:func:`forward_axis` of ``up``; set on creation).
    """

    t: np.ndarray
    up: np.ndarray
    heading: np.ndarray
    heading_smooth: np.ndarray
    meta: dict[str, Any]
    q: np.ndarray | None = None
    fwd: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.fwd = forward_axis(self.up)
        if self.forward_axis != axis_name(FORWARD_V):  # upright: recorded (helmet meta as is)
            self.meta["forward_axis"] = self.forward_axis

    @property
    def forward_axis(self) -> str:
        """Name of :attr:`fwd` (:func:`axis_name`)."""
        return axis_name(self.fwd)

    def cam_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Camera -> world rotations ``(K, 3, 3)`` at frame indices ``idx`` (clamped)."""
        assert self.q is not None
        return quat_matrix(self.q[np.clip(idx, 0, len(self.q) - 1)])

    def index(self, t: float) -> int:
        """Nearest frame index for time ``t``."""
        return int(np.clip(np.round(t * self.meta["fps"]), 0, len(self.t) - 1))

    def r_vw(self, t: float) -> np.ndarray:
        """``R_VW`` at time ``t``: display-world -> camera coordinates."""
        i = self.index(t)
        return level_matrix(self.up[i], self.fwd) @ rot_y(self.heading[i] - self.heading_smooth[i])

    def tilt_deg(self, t: float) -> float:
        """Angle between the camera's ``-z`` (sky lens) and up (for logs)."""
        u = self.up[self.index(t)]
        return math.degrees(math.acos(max(-1.0, min(1.0, -u[2]))))


def _orient_path(source: Path) -> Path:
    """Cache file of a clip's orientation track (meta in the ``.json`` beside it)."""
    return cache.source_dir(source) / f"x3-orient-v{ORIENT_VERSION}.npz"


def _cached_orientation(z: Any, meta: dict[str, Any]) -> Orientation:
    """:class:`Orientation` from the arrays of an open orientation ``.npz``."""
    return Orientation(
        t=z["t"],
        up=z["up"],
        heading=z["heading"],
        heading_smooth=z["heading_smooth"],
        meta=meta,
        q=z["q"],
    )


#: What reading a damaged or half-written ``.npz`` raises.
_NPZ_ERRORS = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile)


def _read_orientation(path: Path, meta: dict[str, Any]) -> Orientation | None:
    """The track cached in ``path``, or None when it is missing or unreadable (corrupt)."""
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return _cached_orientation(z, meta)
    except _NPZ_ERRORS as err:
        logger.debug("%s: orientation cache unreadable (%s); treated as missing", path, err)
        return None


def load_orientation(source: Path) -> Orientation | None:
    """The cached orientation track of a clip, or None (never parses the IMU)."""
    try:
        path = _orient_path(source)
    except OSError:
        return None
    meta = cache.load_json(path.with_suffix(".json")) or {}
    if "fps" not in meta:
        return None
    return _read_orientation(path, meta)


def cached_forward_axis(source: Path) -> str | None:
    """:attr:`Orientation.forward_axis` of a clip's cached track (reads only its ``up``), or
    None when there is none (never parses the IMU)."""
    try:
        path = _orient_path(source)
        with np.load(path) as z:
            return axis_name(forward_axis(z["up"]))
    except _NPZ_ERRORS:
        return None


_AXIS_LOGGED: set[str] = set()  # sources whose non-default forward axis was logged


def wearer_yaw(orient: Orientation, times: np.ndarray, axis: np.ndarray = WEARER_V) -> np.ndarray:
    """Display yaw (deg) of the wearer (``axis`` in camera coordinates) at ``times``."""
    out = np.empty(len(times))
    for i, t in enumerate(times):
        y, _ = ray_yaw_pitch(orient.r_vw(float(t)).T @ axis)
        out[i] = float(y)
    return out


def load_imu(insv00: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t_s, accel m/s^2 (N, 3), gyro deg/s (N, 3))`` in IMU axes, video clock."""
    import telemetry_parser as tp  # noqa: PLC0415

    imu = tp.Parser(str(insv00)).normalized_imu()
    imu = [s for s in imu or [] if s.get("accl") is not None and s.get("gyro") is not None]
    if len(imu) < 100:
        raise ValueError(f"{insv00.name}: no IMU samples")
    t = np.array([s["timestamp_ms"] for s in imu]) / 1000.0
    return (t, np.array([s["accl"] for s in imu], float), np.array([s["gyro"] for s in imu], float))


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(N, 3, 3)`` of unit quaternions ``(N, 4)`` (``w, x, y, z``)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )


def quat_interp(t: np.ndarray, q: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Normalised linear interpolation of a continuous quaternion track (clamped at the ends)."""
    i = np.clip(np.searchsorted(t, times), 1, len(t) - 1)
    f = np.clip((times - t[i - 1]) / (t[i] - t[i - 1]), 0.0, 1.0)[:, None]
    out = (1 - f) * q[i - 1] + f * q[i]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def quat_track(t: np.ndarray, a_v: np.ndarray, w_v: np.ndarray, gain: float) -> np.ndarray:
    """Camera -> world quaternions at every IMU sample (full 3-axis complementary filter).

    Each gyro sample (rad/s, ``V`` axes) turns the camera by ``w dt`` in its own
    frame; the estimated up (world -y seen from the camera) is pulled toward
    the accelerometer as in :func:`fuse`. Unlike the up-vector filter this
    keeps the heading right at any tilt.
    """
    a0 = a_v[:200].mean(0)
    a0 = a0 / (np.linalg.norm(a0) + 1e-12)
    qw, qx, qy, qz = 1.0 - a0[1], a0[2], 0.0, -a0[0]  # shortest turn of a0 onto (0, -1, 0)
    if qw < 1e-6:  # camera upside down: half turn about x
        qw, qx = 0.0, 1.0
    n0 = math.sqrt(qw * qw + qx * qx + qz * qz)
    qw, qx, qz = qw / n0, qx / n0, qz / n0
    tl, al, wl = t.tolist(), a_v.tolist(), w_v.tolist()
    out = []
    prev = tl[0]
    for ti, (ax, ay, az), (wx, wy, wz) in zip(tl, al, wl, strict=True):
        dt = ti - prev
        prev = ti
        if dt > 0:
            wx, wy, wz = wx * dt, wy * dt, wz * dt
            an = math.sqrt(ax * ax + ay * ay + az * az)
            if an > 1e-6:
                ux = -2 * (qx * qy + qw * qz)  # up in camera axes: -(row 1 of R)
                uy = 2 * (qx * qx + qz * qz) - 1
                uz = -2 * (qy * qz - qw * qx)
                dev = (an - G) / (0.15 * G)
                k = gain * dt * math.exp(-dev * dev) / an
                wx -= k * (uy * az - uz * ay)
                wy -= k * (uz * ax - ux * az)
                wz -= k * (ux * ay - uy * ax)
            ang = math.sqrt(wx * wx + wy * wy + wz * wz)
            if ang > 0:
                s = math.sin(ang / 2) / ang
                dw, dx, dy, dz = math.cos(ang / 2), wx * s, wy * s, wz * s
                qw, qx, qy, qz = (
                    qw * dw - qx * dx - qy * dy - qz * dz,
                    qw * dx + qx * dw + qy * dz - qz * dy,
                    qw * dy - qx * dz + qy * dw + qz * dx,
                    qw * dz + qx * dy - qy * dx + qz * dw,
                )
                nq = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                qw, qx, qy, qz = qw / nq, qx / nq, qy / nq, qz / nq
        out.append((qw, qx, qy, qz))
    return np.array(out)


def gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass along axis 0 (``sigma`` in samples, +-3 sigma kernel, edge-padded)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if x.ndim == 1:
        return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")
    xp = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def fuse(
    t: np.ndarray,
    acc: np.ndarray,
    gyr: np.ndarray,
    fps: float,
    n_frames: int,
    rate_hz: float = 200.0,
    gain: float = 0.6,
    heading_smooth_s: float = 1.5,
    q_offset_s: float = 0.0,
) -> Orientation:
    """Complementary filter: gyro-propagated gravity, pulled toward the accelerometer.

    The up vector in camera coordinates follows ``du/dt = -w x u`` (gyro, rad/s)
    and is nudged toward the accelerometer direction with ``gain`` per second,
    weighted down when ``|a|`` is far from 1 g (skiing accelerations). The
    heading integrates the gyro component about the vertical. The full
    rotation (``q``) comes from :func:`quat_track` at the IMU rate.

    Args:
        t, acc, gyr: IMU samples (IMU axes, m/s^2 and deg/s).
        fps: Video frame rate.
        n_frames: Number of video frames.
        rate_hz: Filter rate (the ~1 kHz IMU is averaged down to this).
        gain: Accelerometer correction (1/s).
        heading_smooth_s: Gaussian sigma (s) of the display heading.
        q_offset_s: ``q`` is sampled this long after each frame time (IMU sync).
    """
    a_v = acc @ M_IMU_TO_V.T
    w_v = np.radians(gyr @ M_IMU_TO_V.T)
    # Average onto a regular grid.
    dt = 1.0 / rate_hz
    t0 = float(t[0])
    idx = np.floor((t - t0) / dt).astype(int)
    nb = int(idx[-1]) + 1
    cnt = np.bincount(idx, minlength=nb).astype(float)
    ok = cnt > 0

    def bin3(x: np.ndarray) -> np.ndarray:
        out = np.stack([np.bincount(idx, weights=x[:, k], minlength=nb) for k in range(3)], 1)
        out[ok] /= cnt[ok, None]
        for k in range(3):  # fill empty bins
            out[~ok, k] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), out[ok, k])
        return out

    ab, wb = bin3(a_v), bin3(w_v)
    tb = t0 + (np.arange(nb) + 0.5) * dt
    a0 = ab[: max(1, int(0.5 * rate_hz))].mean(0)
    ux, uy, uz = (a0 / (np.linalg.norm(a0) + 1e-12)).tolist()
    up = np.empty((nb, 3))
    head = np.empty(nb)
    psi = 0.0
    abl, wbl = ab.tolist(), wb.tolist()
    for i in range(nb):
        wx, wy, wz = wbl[i]
        # u <- u - (w x u) dt
        cx, cy, cz = wy * uz - wz * uy, wz * ux - wx * uz, wx * uy - wy * ux
        ux, uy, uz = ux - cx * dt, uy - cy * dt, uz - cz * dt
        ax, ay, az = abl[i]
        an = math.sqrt(ax * ax + ay * ay + az * az)
        if an > 1e-6:
            dev = (an - G) / (0.15 * G)
            k = gain * dt * math.exp(-dev * dev)
            ux += k * (ax / an - ux)
            uy += k * (ay / an - uy)
            uz += k * (az / an - uz)
        nu = math.sqrt(ux * ux + uy * uy + uz * uz)
        ux, uy, uz = ux / nu, uy / nu, uz / nu
        psi += (wx * ux + wy * uy + wz * uz) * dt
        up[i] = (ux, uy, uz)
        head[i] = psi
    ft = np.arange(n_frames) / fps
    upf = np.stack([np.interp(ft, tb, up[:, k]) for k in range(3)], 1)
    upf /= np.linalg.norm(upf, axis=1, keepdims=True)
    hf = np.degrees(np.interp(ft, tb, head))
    hs = gauss_smooth(hf, max(1, int(heading_smooth_s * fps)))
    q = quat_interp(t, quat_track(t, a_v, w_v, gain), ft + q_offset_s)
    return Orientation(
        t=ft,
        up=upf,
        heading=hf,
        heading_smooth=hs,
        q=q,
        meta={
            "fps": fps,
            "rate_hz": rate_hz,
            "gain": gain,
            "heading_smooth_s": heading_smooth_s,
            "q_offset_s": q_offset_s,
            "imu_hz": round((len(t) - 1) / max(t[-1] - t[0], 1e-6), 1),
            "imu_t0": round(t0, 3),
        },
    )


def orientation(
    source: Path,
    insv00: Path,
    fps: float,
    n_frames: int,
    force: bool = False,
    q_offset_s: float | None = None,
) -> Orientation:
    """Cached :func:`fuse` for one clip (``cache/<lrv>/x3-orient-v2.npz``).

    ``q_offset_s`` (the renderer's IMU sync) recomputes a track cached with
    another offset; None takes any. An unreadable (corrupt) cache file is computed anew. The
    ``.npz`` is written atomically (:func:`cutpoint.cache.save_bytes`), so a concurrent
    reader never sees half of it.
    """
    path = _orient_path(source)
    meta = cache.load_json(path.with_suffix(".json")) or {"fps": fps}
    same = q_offset_s is None or meta.get("q_offset_s", 0.0) == q_offset_s
    if not force and same and (cached := _read_orientation(path, meta)) is not None:
        _log_axis(source, cached)
        return cached
    t0 = time.perf_counter()
    t, acc, gyr = load_imu(insv00)
    o = fuse(t, acc, gyr, fps, n_frames, q_offset_s=q_offset_s or 0.0)
    o.meta["elapsed_s"] = round(time.perf_counter() - t0, 2)
    buf = io.BytesIO()  # a file object: numpy adds no suffix
    np.savez_compressed(
        buf,
        t=o.t.astype(np.float32),
        up=o.up.astype(np.float32),
        heading=o.heading.astype(np.float32),
        heading_smooth=o.heading_smooth.astype(np.float32),
        q=o.q.astype(np.float32),
    )
    cache.save_bytes(path, buf.getvalue())
    cache.save_json(path.with_suffix(".json"), o.meta)
    _log_axis(source, o)
    return o


def _log_axis(source: Path, o: Orientation) -> None:
    """INFO once per source and process when its yaw 0 is not :data:`FORWARD_V`."""
    if o.forward_axis != axis_name(FORWARD_V) and str(source) not in _AXIS_LOGGED:
        _AXIS_LOGGED.add(str(source))
        logger.info(
            "%s: camera upright (lenses horizontal); yaw 0 follows the _10_ lens axis (%s)",
            Path(source).name,
            o.forward_axis,
        )


# --------------------------------------------------------------------------- projection
def flat_rays(width: int, height: int, h_fov: float, v_fov: float) -> np.ndarray:
    """Unit rays ``(H, W, 3)`` of a rectilinear view (x right, y down, z forward)."""
    tx, ty = math.tan(math.radians(h_fov) / 2), math.tan(math.radians(v_fov) / 2)
    xs = ((2 * np.arange(width) + 1) / width - 1) * tx
    ys = ((2 * np.arange(height) + 1) / height - 1) * ty
    x, y = np.meshgrid(xs, ys)
    d = np.stack([x, y, np.ones_like(x)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def ray_yaw_pitch(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Display yaw (deg, + = right) and pitch (deg, + = up) of world rays."""
    yaw = np.degrees(np.arctan2(d[..., 0], d[..., 2]))
    pitch = np.degrees(np.arcsin(np.clip(-d[..., 1], -1, 1)))
    return yaw, pitch


def yaw_pitch_ray(yaw: float | np.ndarray, pitch: float | np.ndarray) -> np.ndarray:
    """Inverse of :func:`ray_yaw_pitch`."""
    y, p = np.radians(yaw), np.radians(pitch)
    return np.stack([np.cos(p) * np.sin(y), -np.sin(p), np.cos(p) * np.cos(y)], -1)


POLAR_LINE_MAX_DEG = 80.0  # the yaw line stays this far from the poles (keeps a width)
FRONT_Z = 0.1  # visible share: points within ~84 deg of the view axis (others blow up)


def _view_xy(
    r: np.ndarray, tx: float, ty: float, yaws: np.ndarray, pitches: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised view coordinates (-1..1 inside) and ``z`` of display rays."""
    d = yaw_pitch_ray(yaws, np.clip(pitches, -90.0, 90.0)) @ r  # rows: r.T @ ray
    z = np.maximum(d[..., 2], 1e-3)
    x = np.clip(d[..., 0] / z / tx, -1e3, 1e3)
    y = np.clip(d[..., 1] / z / ty, -1e3, 1e3)
    return x, y, d[..., 2]


def project_box_visible(
    view_yaw: float,
    view_pitch: float,
    hfov: float,
    out_w: int,
    out_h: int,
    box_yaw: float,
    box_pitch: float,
    box_w: float,
    box_h: float,
    steps: int = 9,
    polar_shift: float = 0.5,
) -> tuple[tuple[float, float, float, float], float] | None:
    """Where a display-frame box lands in a rendered flat view (fractions of the frame).

    The view is the renderer's: output ray ``d`` of a rectilinear frame
    (:func:`flat_rays`, horizontal FOV ``hfov``, vertical :func:`flat_fov`) looks
    along ``view_matrix(view_yaw, view_pitch) @ d``. The box is an equirect
    rectangle (the view layer's detections): centre ``box_yaw`` / ``box_pitch``,
    ``box_w`` deg of yaw by ``box_h`` deg of pitch. A cross of ``steps``-point
    lines is projected and bounded: a vertical line, and the yaw span at
    ``polar_shift`` of the way from the centre to the edge nearer the pole
    (at most :data:`POLAR_LINE_MAX_DEG` from the horizon). A person's equirect
    width is set by its most stretched (most polar) part, so the full rectangle
    inflates the box. Measured against the detector on 16 rendered clips (~860
    matched boxes): rectangle 1.20x too wide; centre line unbiased below 0.5 of
    the frame but 1.20x wide above it (close friends on a chairlift);
    ``polar_shift`` 0.5 gives 0.89-1.02x wide and 1.00-1.03x tall at every size
    and the best per-clip QA agreement.

    The vertical line is the box's centre line, or, when that is outside the
    frame, the visible vertical line nearest to it (so a box cut by the frame
    edge gets the height it has at the edge). ``y1`` / ``y2`` bound the points
    with ``|x| <= 1`` (all points when none qualify), so a part far beside the
    frame (or behind the camera, where points are pushed far out) does not
    stretch the height; ``x1`` / ``x2`` bound all points (bounding x by the
    points with ``|y| <= 1`` too, or moving an off-frame yaw line into the
    frame, tracked the rendered QA worse: score r 0.80 vs 0.86).

    The share of the box inside the frame (0-1) is the clipped box's area over
    the box's projected extent (both in frame units, the extent clamped to 2
    frame widths / heights beyond the edges; its height also counts the centre
    cross's points within ~84 deg of the view axis, :data:`FRONT_Z`, so parts
    beside the frame enlarge it but points behind the camera do not).
    An angular share (fraction of a yaw / pitch grid inside the frame) was
    tried and tracked the rendered detector worse (score r 0.74 vs 0.84).

    Returns:
        ``((x1, y1, x2, y2), share)``: the box with 0-1 inside the frame (x right,
        y down, not clipped) and its visible share, or None when the box centre
        is behind the camera.
    """
    vfov = flat_fov(hfov, out_w, out_h)
    tx, ty = math.tan(math.radians(hfov) / 2), math.tan(math.radians(vfov) / 2)
    r = view_matrix(view_yaw, view_pitch)
    c = r.T @ yaw_pitch_ray(box_yaw, box_pitch)
    if c[2] <= 1e-6:
        return None
    s = np.linspace(-0.5, 0.5, steps)
    lim = POLAR_LINE_MAX_DEG
    p_w = box_pitch + math.copysign(polar_shift * box_h / 2, box_pitch)
    p_w = min(lim, max(-lim, p_w))
    # vertical lines at every grid yaw: (steps yaws, steps pitches)
    gy, gp = np.meshgrid(box_yaw + s * box_w, box_pitch + s * box_h, indexing="ij")
    vx, vy, vz = _view_xy(r, tx, ty, gy, gp)
    centre = steps // 2
    line = centre
    if not (np.abs(vx[centre]) <= 1).any():
        vis = [k for k in range(steps) if (np.abs(vx[k]) <= 1).any()]
        if vis:
            line = min(vis, key=lambda k: abs(k - centre))
    hx, hy, hz = _view_xy(r, tx, ty, box_yaw + s * box_w, np.full(steps, p_w))
    xs = np.r_[hx, vx[line]]
    y = np.r_[hy, vy[line]]
    in_x = np.abs(xs) <= 1
    ys = y[in_x] if in_x.any() else y
    box = (
        float((xs.min() + 1) / 2),
        float((ys.min() + 1) / 2),
        float((xs.max() + 1) / 2),
        float((ys.max() + 1) / 2),
    )
    # visible share: the clipped box over the box's whole extent (clamped)
    front = np.r_[hz, vz[centre]] > FRONT_Z
    fy = np.r_[np.r_[hy, vy[centre]][front], ys]
    full = np.clip(np.r_[xs.min(), fy.min(), xs.max(), fy.max()], -3.0, 3.0)
    full_area = (full[2] - full[0]) * (full[3] - full[1])
    vis = np.clip([xs.min(), ys.min(), xs.max(), ys.max()], -1.0, 1.0)
    vis_area = max(0.0, vis[2] - vis[0]) * max(0.0, vis[3] - vis[1])
    share = float(vis_area / full_area) if full_area > 0 else 0.0
    return box, share


class FisheyeSampler:
    """GPU sampler: perspective / equirect views from dual-fisheye frames.

    Rays are built once per view layout; per frame only the 3x3 ``R_VW``
    changes, so each batch is one matrix multiply plus ``grid_sample``.
    """

    def __init__(self, device: str, fov: float = DFISHEYE_FOV) -> None:
        import torch  # noqa: PLC0415

        self.torch, self.device, self.fov = torch, device, fov

    def rays(self, d: np.ndarray) -> Any:
        """Upload world rays ``(..., 3)`` once."""
        return self.torch.as_tensor(
            d.reshape(-1, 3), dtype=self.torch.float32, device=self.device
        ), d.shape[:-1]

    def sample(
        self,
        frames: Any,
        r_vw: np.ndarray,
        rays: tuple[Any, tuple[int, ...]],
        view_rot: np.ndarray | None = None,
    ) -> Any:
        """Sample ``frames`` (B, 3, H, W float 0-1) along rays.

        Args:
            frames: Dual-fisheye frames on the GPU.
            r_vw: ``(B, 3, 3)`` display-world -> camera rotations.
            rays: From :meth:`rays`, shape ``(*S)``.
            view_rot: Optional ``(K, 3, 3)`` view rotations (world <- view); the
                result then has K views per frame.

        Returns:
            ``(B, K, 3, *S)`` (or ``(B, 3, *S)`` without ``view_rot``).
        """
        torch = self.torch
        d, shape = rays
        b, _, _h, _w = frames.shape
        rv = torch.as_tensor(r_vw, dtype=torch.float32, device=self.device)
        if view_rot is not None:
            vr = torch.as_tensor(view_rot, dtype=torch.float32, device=self.device)
            full = torch.einsum("bij,kjl->bkil", rv, vr)  # (B, K, 3, 3)
        else:
            full = rv[:, None]
        k = full.shape[1]
        cam = torch.einsum("bkij,nj->bkni", full, d)  # (B, K, N, 3)
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        hh = torch.sqrt(x * x + y * y).clamp_min(1e-9)
        theta = torch.arccos(z.abs().clamp(max=1.0))
        r = theta / math.radians(self.fov / 2)
        uf = (r * x / hh) * 0.25  # in units of the full width, centred per half
        vf = (r * y / hh) * 0.5
        u = torch.where(z >= 0, 0.75 + uf, 0.25 - uf)
        v = 0.5 + vf
        grid = torch.stack([u * 2 - 1, v * 2 - 1], -1).view(b, k * d.shape[0], 1, 2)
        out = torch.nn.functional.grid_sample(
            frames, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        out = out.view(b, 3, k, *shape).permute(0, 2, 1, *range(3, 3 + len(shape)))
        return out if view_rot is not None else out[:, 0]
