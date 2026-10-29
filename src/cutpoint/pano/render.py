"""Render a director path to a flat MP4 with ffmpeg ``v360`` + ``sendcmd``.

For every update time the virtual camera (display-frame yaw / pitch / FOV from
the path) is composed with the clip's horizon-levelling rotation and written
as ``v360`` commands (``rorder=ryp``, see :mod:`cutpoint.pano.geometry`).
Runtime rotation commands are *incremental* in ffmpeg 9.0.2 (measured: each
``yaw`` / ``pitch`` / ``roll`` command right-multiplies the current rotation
by that one axis, in the order listed; re-sending ``yaw 30`` turns another 30
deg), so each update sends the delta from the tracked current rotation::

    0.0667 v360@cam roll -0.412, v360@cam yaw 0.087, v360@cam pitch 0.233;

Each changed ``v360`` option rebuilds its remap table (~33 ms at 1920x1080 on
this machine), but stabilisation needs the correction on every frame: a
command sent every second frame and held leaves the shake visible on every
other frame (measured: 40-64% of the rendered motion at 15 Hz). So one
command goes out per frame (``cmd_every``, default 1), timed just before that
frame's timestamp, and an angle is only re-sent when it moved by more than
0.05 deg. Measured on a 10 s 5.7K clip: no commands 5.7 s, one command per
frame 15.4 s, three per frame 37 s.

Stabilisation (unlevelled sources): the camera's full rotation
(:attr:`~cutpoint.pano.geometry.Orientation.q`) is undone per frame and
replaced by a display frame whose yaw 0 is the smoothed heading and whose up
is the true vertical (``level_strength`` 1) or follows the camera's own
smoothed tilt (0; ``tilt_follow_s``), blended in between. The tilt followed is
relative to the clip's mean attitude (a fixed mount cancels out) and capped at
``max_follow_deg`` before the blend, so a fall or the camera being handled
(a turn of 100 deg or more) shows at most ``(1 - level_strength) * max_follow_deg``.

Sources (:func:`cutpoint.pano.project.resolve_source`): the raw dual fisheye
(``_00_`` + ``_10_`` stacked, or the 4K side-by-side file) or an SDK equirect
stitched without FlowState, both levelled with our own IMU orientation (the
equirect through the constant
:data:`~cutpoint.pano.geometry.SDK_EQUIRECT_FROM_CAMERA`); or a ``levelled``
equirect (FlowState SDK stitch, Studio export), on which the path angles are
used directly although its yaw 0 does not follow the director's display
frame (see :func:`~cutpoint.pano.project.resolve_source`). Audio is the clip's original track,
taken from the ``.lrv`` (the same AAC stream as ``_00_.insv``) with the
usual baked fade.

Clip export integrates with ``cutpoint clips`` / ``cutpoint extend`` through
:class:`PanoExporter`: numbering, tails, padding, speech-aware boundaries and
the manifest work as for any other source; the manifest entry gets a
``pano`` record (clip, source mode, path file, preview flag).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.encode import A_FMT, BT709_TAGS, Grade, audio_fade_filter, bt709_filter
from cutpoint.ffmpeg_utils import (
    READ_ERROR_RE,
    FFmpegError,
    HwDecoder,
    VideoEncoder,
    atomic_output,
    decoder_for,
    encode_with_fallback,
    gpu_chain,
    output_dir,
    probe,
    run,
)
from cutpoint.pano import director as D
from cutpoint.pano import geometry as G
from cutpoint.pano.project import (
    RenderSource,
    clip_for,
    load,
    pano_config,
    resolve_source,
    source_flowstate,
)
from cutpoint.pano.x3 import X3Clip, imu_source

logger = logging.getLogger(__name__)

PATHS_DIR = "pano_paths"
ANGLE_EPS = 0.05
FOV_EPS = 0.2
CMD_LEAD_S = 0.005  # commands go out this long before the frame they are meant for
# First line of a ``.cmd`` file, a sendcmd comment: ``input=``, the initial ``rorder``,
# ``yaw``, ``pitch`` and ``roll``, the seek ``start`` and the source ``fps`` (for ``360 qa``)
CMD_HEADER = "# cutpoint-v360"
MAX_FOLLOW_DEG = 30.0  # default ``pano.max_follow_deg``
PROGRESS_POLL_S = 0.1  # ``render_segment(progress=)``: ffmpeg's progress file is read this often
PROGRESS_MIN_S = 0.1  # ... and the callback called at most once per this many seconds
UPRIGHT = "upright"  # path ``level``: no IMU, level for the nominal mount (sky lens up)


@dataclass(frozen=True)
class Format360:
    """Output geometry."""

    width: int = 1920
    height: int = 1080
    fps: int = 30

    @property
    def vertical(self) -> bool:
        """Portrait output."""
        return self.height > self.width


DEFAULT_FORMAT = Format360()


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass along axis 0 (edge-padded)."""
    if sigma <= 0.5:
        return x
    return G.gauss_smooth(x, sigma)


def _slerp(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    """Spherical interpolation from unit vectors ``a`` to ``b`` (rows) by ``f``."""
    om = np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))[:, None]
    so = np.sin(om)
    safe = np.where(so > 1e-6, so, 1.0)
    out = np.where(so > 1e-6, (np.sin((1 - f) * om) * a + np.sin(f * om) * b) / safe, a)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def stabilised_r_vw(
    orient: G.Orientation,
    idx: np.ndarray,
    level_strength: float,
    tilt_follow_s: float,
    max_follow_deg: float = MAX_FOLLOW_DEG,
) -> np.ndarray:
    """``R_VW`` (display -> camera) ``(K, 3, 3)`` for the increasing frame indices ``idx``.

    The display frame (display -> world) has its yaw 0 at the heading of the
    clip's forward axis (``orient.fwd``) low-passed over ``heading_smooth_s``
    (as the director's frame) and its up
    ``level_strength`` of the way from the camera's own up axis (its mean over
    ``idx``, as the world sees it, low-passed over ``tilt_follow_s``, and pulled
    back along its great circle to at most ``max_follow_deg`` from vertical) to
    the true vertical. ``R_VW`` is the camera rotation transposed times that frame.
    """
    fps = float(orient.meta["fps"])
    sigma = float(orient.meta.get("heading_smooth_s", 1.5)) * fps
    lo = max(0, int(idx[0]) - int(3 * sigma) - 1)
    hi = min(len(orient.q), int(idx[-1]) + int(3 * sigma) + 2)
    rel = np.clip(idx - lo, 0, hi - lo - 1)
    rw = orient.cam_to_world(np.arange(lo, hi))  # camera -> world
    fw = rw @ orient.fwd
    heading = np.unwrap(np.arctan2(fw[:, 0], fw[:, 2]))
    hs = _gauss(heading, sigma)[rel]
    up_w = np.array([0.0, -1.0, 0.0])
    nominal = np.einsum("kji,j->ki", rw[rel], up_w).mean(0)  # mean up in camera axes
    nominal /= np.linalg.norm(nominal)
    own = _gauss(rw @ nominal, tilt_follow_s * fps)[rel]
    own /= np.linalg.norm(own, axis=1, keepdims=True)
    f = np.stack([np.sin(hs), np.zeros_like(hs), np.cos(hs)], 1)
    c = own @ up_w
    cap = math.radians(max(float(max_follow_deg), 0.0))
    side = own - c[:, None] * up_w  # direction of the lean, on the great circle to vertical
    n = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(n > 1e-6, side / np.maximum(n, 1e-12), f)  # upside down: lean forward
    capped = math.cos(cap) * up_w + math.sin(cap) * side
    own = np.where((np.arccos(np.clip(c, -1.0, 1.0)) > cap)[:, None], capped, own)
    u = _slerp(own, np.tile(up_w, (len(rel), 1)), level_strength)
    y = -u
    f -= (f * y).sum(1, keepdims=True) * y
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    disp = np.stack([np.cross(y, f), y, f], 2)  # columns: display axes in world
    return np.einsum("kji,kjl->kil", rw[rel], disp)


def clip_orientation(clip: X3Clip) -> G.Orientation:
    """The IMU orientation track :func:`render_segment` levels ``clip`` with (cached; with
    ``pano.imu_offset_s``).

    Raises:
        OSError: If the source is not reachable.
        Exception: Any IMU read error when the track is not cached yet (no ``_00_.insv``,
            no IMU samples, an unreadable trailer).
    """
    src = Path(clip.source)
    return G.orientation(
        src,
        imu_source(src),
        clip.fps,
        max(clip.frames, 1),
        q_offset_s=float(pano_config().get("imu_offset_s", 0.0)),
    )


def upright_orientation(fps: float, n_frames: int) -> G.Orientation:
    """A still camera in the nominal mount (sky lens up, ``up`` = camera ``-z``) for
    ``n_frames`` frames: the horizon of a clip without IMU data (path ``level``
    :data:`UPRIGHT`). Fused by :func:`~cutpoint.pano.geometry.fuse` from a synthetic
    resting IMU, so its axes and quaternion follow the same conventions as a real track."""
    t = np.arange(0.0, 1.0, 1e-3)
    acc = np.tile(G.M_IMU_TO_V @ np.array([0.0, 0.0, -G.G]), (len(t), 1))  # gravity: rest
    one = G.fuse(t, acc, np.zeros((len(t), 3)), fps, 1)
    n = max(int(n_frames), 1)
    return G.Orientation(
        t=np.arange(n) / fps,
        up=np.tile(one.up[0], (n, 1)),
        heading=np.zeros(n),
        heading_smooth=np.zeros(n),
        meta=one.meta | {"level": UPRIGHT},
        q=np.tile(one.q[0], (n, 1)),
    )


class _Progress:
    """Forwards the ``-progress`` output of :func:`render_segment`'s ffmpeg runs to a callback.

    Each ffmpeg run writes its own temp file (:meth:`attempt`), read by a helper thread every
    :data:`PROGRESS_POLL_S`. Values are ``out_time / dur``, capped below 1.0, sent only when
    they rise (a GPU failure redone on the CPU starts over: the fraction holds until it passes
    the earlier one) and at most once per :data:`PROGRESS_MIN_S`; :meth:`done` sends the one
    1.0. Calls never overlap (a lock). The first exception of the callback is logged at DEBUG
    and the callback is dropped.
    """

    def __init__(self, fn: Callable[[float], object], total: float, name: str) -> None:
        self.fn: Callable[[float], object] | None = fn
        self.total = total
        self.name = name
        self.last = 0.0
        self.sent = -math.inf
        self.lock = threading.Lock()

    def emit(self, frac: float, final: bool = False) -> None:
        with self.lock:
            if self.fn is None:
                return
            if not final:
                frac = min(max(frac, 0.0), 0.999)
                now = time.monotonic()
                if frac <= self.last or now - self.sent < PROGRESS_MIN_S:
                    return
                self.sent = now
            self.last = frac
            try:
                self.fn(frac)
            except Exception:  # a progress callback never breaks the render
                logger.debug(
                    "%s: progress callback failed; not called again", self.name, exc_info=True
                )
                self.fn = None

    def done(self) -> None:
        self.emit(1.0, final=True)

    def _tail(self, file: Path, stop: threading.Event) -> None:
        buf = ""
        try:
            with file.open(encoding="utf-8", errors="replace") as fh:
                while True:
                    stopped = stop.wait(PROGRESS_POLL_S)
                    buf += fh.read()
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        if not line.startswith("out_time_us=") or self.total <= 0:
                            continue
                        with contextlib.suppress(ValueError):  # "N/A" before the first frame
                            self.emit(int(line.split("=", 1)[1]) / 1e6 / self.total)
                    if stopped:
                        return
        except OSError as err:
            logger.debug("%s: progress not read (%s)", self.name, err)

    @contextlib.contextmanager
    def attempt(self) -> Any:
        """A fresh progress file for one ffmpeg run (yielded), read while the run lasts;
        the reader is joined and the file removed on exit."""
        fd, name = tempfile.mkstemp(prefix="cutpoint-progress-", suffix=".txt")
        os.close(fd)
        file = Path(name)
        stop = threading.Event()
        reader = threading.Thread(
            target=self._tail, args=(file, stop), name="pano-render-progress", daemon=True
        )
        reader.start()
        try:
            yield file
        finally:
            stop.set()
            reader.join()
            try:  # never fails the render (or hides a Ctrl+C): a scanner may hold it on Windows
                file.unlink(missing_ok=True)
            except OSError as err:
                logger.debug("%s: progress file %s not removed (%s)", self.name, file, err)


def command_lines(
    clip: X3Clip,
    path: dict[str, Any],
    src: RenderSource,
    start: float,
    dur: float,
    fmt: Format360,
    cmd_every: int,
    orient: G.Orientation | None,
) -> tuple[list[str], dict[str, float]]:
    """``sendcmd`` lines (relative times) and the initial ``v360`` option values.

    The ``v360`` rotation maps output rays to input rays: ``view`` for a
    ``levelled`` source, ``R_VW @ view`` for the dual fisheye and
    ``SDK_EQUIRECT_FROM_CAMERA @ R_VW @ view`` for an unlevelled equirect,
    with ``R_VW`` from :func:`stabilised_r_vw` at the displayed frame. After
    the seek the first frame is source frame ``ceil(start * fps)``, so each
    command is timed :data:`CMD_LEAD_S` before its frame's timestamp.
    """
    cfg = pano_config()
    fps = clip.fps or 30000 / 1001
    j0 = math.ceil(start * fps - 1e-6)
    n = math.ceil(dur * fps) + 1  # one spare: the output can end on a frame past start + dur
    idx = j0 + np.arange(0, n, max(1, cmd_every))
    rel = np.maximum(idx / fps - start - CMD_LEAD_S, 0.0)
    times = idx / fps
    yaw, pitch, fov = D.evaluate(path, times)
    yaw_off = float(cfg.get("equirect_yaw_offset", 0.0))
    r_vw = None
    if not src.levelled and orient is not None:
        r_vw = stabilised_r_vw(
            orient,
            idx,
            float(cfg.get("level_strength", 0.7)),
            float(cfg.get("tilt_follow_s", 0.5)),
            float(cfg.get("max_follow_deg", MAX_FOLLOW_DEG)),
        )
    lines: list[str] = []
    init: dict[str, float] = {}
    cur = np.eye(3)  # rotation v360 currently applies (tracked exactly as sent)
    last_fov = (0.0, 0.0)
    for i, t in enumerate(rel):
        view = G.view_matrix(float(yaw[i]) + (yaw_off if src.levelled else 0.0), float(pitch[i]))
        if r_vw is not None:
            r = r_vw[i] @ view
            if src.projection == "equirect":  # SDK stitch without FlowState
                r = G.SDK_EQUIRECT_FROM_CAMERA @ r
        else:
            r = view
        h = float(fov[i])
        v = G.flat_fov(h, fmt.width, fmt.height)
        if i == 0:
            y_, p_, r_ = (round(x, 3) for x in G.decompose_ryp(r))
            init = {"yaw": y_, "pitch": p_, "roll": r_, "h_fov": h, "v_fov": v}
            cur = G.rot_r(r_) @ G.rot_y(y_) @ G.rot_p(p_)
            last_fov = (h, v)
            continue
        # Runtime yaw/pitch/roll commands are *incremental*: each one right-multiplies
        # the current rotation by that single-axis rotation, in the order listed
        # (measured on ffmpeg 9.0.2). So send the delta cur^T @ target as
        # roll, yaw, pitch (the ryp decomposition), and track what was sent.
        d_y, d_p, d_r = G.decompose_ryp(cur.T @ r)
        parts = []
        for name, val, axis in (
            ("roll", d_r, G.rot_r),
            ("yaw", d_y, G.rot_y),
            ("pitch", d_p, G.rot_p),
        ):
            val = round(val, 3)
            if abs(val) > ANGLE_EPS:
                parts.append(f"v360@cam {name} {val:.3f}")
                cur = cur @ axis(val)
        if abs(h - last_fov[0]) > FOV_EPS:
            parts += [f"v360@cam h_fov {h:.3f}", f"v360@cam v_fov {v:.3f}"]
            last_fov = (h, v)
        if parts:
            lines.append(f"{t:.4f} " + ", ".join(parts) + ";")
    return lines, init


def render_segment(
    clip: X3Clip,
    path: dict[str, Any],
    start: float,
    end: float,
    out: Path,
    *,
    fmt: Format360 = DEFAULT_FORMAT,
    src: RenderSource | None = None,
    audio_fade: float = 0.25,
    cmd_every: int | None = None,
    work_dir: Path | None = None,
    grade: Grade | None = None,
    progress: Callable[[float], object] | None = None,
) -> dict[str, Any]:
    """Render ``[start, end]`` of ``clip`` along ``path`` into ``out``; ``grade`` grades it
    after the BT.709 conversion, as in a flat export.

    Unlevelled sources are levelled with :func:`clip_orientation`, or with
    :func:`upright_orientation` when the path's ``level`` is :data:`UPRIGHT` (a
    ``level_forward`` preview of a clip without IMU data,
    :func:`cutpoint.pano.candidates.preview_path`).

    ``progress``: called with the rendered fraction (0.0 to 1.0) from ffmpeg's ``-progress``
    output: rising values below 1.0 at most about 10 times per second (from a helper
    thread; calls never overlap, so keep it quick), then exactly one 1.0 from the calling
    thread once ``out`` is complete (a failed render sends no 1.0). Never called after this
    function returns or raises. If it raises, that is logged once at DEBUG and it is not called
    again; the render goes on. None: no progress file, the ffmpeg command is unchanged.

    Returns:
        A record: ``mode``, ``inputs``, ``levelled`` (the path angles used on the input as
        is, :class:`~cutpoint.pano.project.RenderSource`), ``flowstate``
        (:func:`~cutpoint.pano.project.source_flowstate`), ``preview``, ``seconds``,
        ``commands``, ``read_error`` (or None).

    Raises:
        FFmpegError: If ffmpeg fails with and without GPU decoding.
    """
    from cutpoint.config import remap_source  # noqa: PLC0415

    src = src or resolve_source(clip)
    cmd_every = int(cmd_every or pano_config().get("cmd_every", 1))
    # Start on a frame, so the audio seek matches the first video frame.
    fps = clip.fps or 30000 / 1001
    start = math.ceil(start * fps - 1e-6) / fps
    dur = end - start
    orient = None
    if not src.levelled:
        if path.get("level") == UPRIGHT:
            orient = upright_orientation(fps, max(clip.frames, math.ceil(end * fps) + 2))
        else:
            orient = clip_orientation(clip)
    lines, init = command_lines(clip, path, src, start, dur, fmt, cmd_every, orient)
    work = work_dir or out.parent
    work.mkdir(parents=True, exist_ok=True)
    # A still path has no commands: ffmpeg rejects an empty sendcmd file, and the
    # initial v360 options already hold the static view.
    rot = f"rorder=ryp:yaw={init['yaw']:.3f}:pitch={init['pitch']:.3f}:roll={init['roll']:.3f}"
    cmd_file = work / (out.stem + ".cmd") if lines else None
    if cmd_file is not None:
        # A sendcmd comment first: the initial rotation and the timing the incremental
        # commands start from, so `360 qa` can replay what ffmpeg was told (CMD_HEADER).
        head = f"{CMD_HEADER} input={src.projection} {rot.replace(':', ' ')} "
        head += f"start={start:.6f} fps={fps:.6f}"
        cmd_file.write_text("\n".join([head, *lines]) + "\n", encoding="utf-8")
    else:  # an earlier render's commands would mislead `360 qa`
        (work / (out.stem + ".cmd")).unlink(missing_ok=True)
    sendcmd = f"sendcmd=f={cmd_file.name}," if cmd_file is not None else ""
    seek = ["-ss", f"{start:.6f}", "-t", f"{dur:.3f}"]
    two = src.projection == "dfisheye" and len(src.inputs) == 2
    if src.projection == "dfisheye":
        v360_in = f"input=dfisheye:ih_fov={G.DFISHEYE_FOV:g}:iv_fov={G.DFISHEYE_FOV:g}"
    else:
        v360_in = "input=equirect"
    audio_idx = len(src.inputs)
    info = probe(remap_source(clip.source))
    audio_end = info.audio_duration
    audio = None if audio_end is None else min(end, audio_end) - start
    a_fmt = A_FMT + audio_fade_filter(dur, audio_fade, audio)
    # HDR / BT.2020 footage is converted to SDR Rec.709 before the 8-bit format (#289), BT.601
    # to BT.709 and untagged HD tagged bt709 (#290), as in a flat export. Insta360 X3 files are
    # tagged BT.709 (full range), so for them nothing changes.
    same = Path(src.inputs[0]) == Path(clip.source)
    color = bt709_filter(info if same else probe(remap_source(src.inputs[0])))
    chain = ",".join(f for f in (color, grade.filter() if grade else "") if f)
    v360 = (
        f"v360@cam={v360_in}:output=flat:{rot}:h_fov={init['h_fov']:.3f}:"
        f"v_fov={init['v_fov']:.3f}:w={fmt.width}:h={fmt.height}:interp=line"
    )
    body = (
        f"{sendcmd}{v360},setpts=PTS-STARTPTS,scale=in_range=pc:out_range=tv,"
        f"setsar=1,fps={fmt.fps}:round=down,{chain + ',' if chain else ''}format=yuv420p"
    )
    af = f"[{audio_idx}:a:0]{a_fmt}[a]"

    def stack_for(dec: HwDecoder | None) -> str:
        """The inputs joined for v360; VAAPI frames are downloaded first (CUDA and CPU ones are not)."""
        dl = gpu_chain(dec, None)
        if not two:
            return f"[0:v]{dl}," if dl else "[0:v]"
        if not dl:
            return "[0:v][1:v]hstack=inputs=2,"
        return f"[0:v]{dl}[d0];[1:v]{dl}[d1];[d0][d1]hstack=inputs=2,"

    def args_for(
        enc: VideoEncoder, dec: HwDecoder | None, dst: Path, prog: Path | None = None
    ) -> list[str]:
        args = ["ffmpeg", "-y", "-v", "error"]
        if prog is not None:
            args += ["-progress", str(prog), "-nostats"]
        # Read moved footage through source_remap; the record keeps the original inputs.
        for inp in src.inputs:
            args += (dec.input_args() if dec else []) + seek + ["-i", str(remap_source(inp))]
        args += [*seek, "-i", str(remap_source(clip.source))]
        return [
            *args,
            "-filter_complex",
            f"{stack_for(dec)}{enc.chain(body)}[v];{af}",
            "-map",
            "[v]",
            "-map",
            "[a]",
            *enc.args(19),
            "-color_range",
            "tv",
            *(BT709_TAGS if color else []),
            "-fps_mode",
            "cfr",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            f"{dur:.3f}",
            "-video_track_timescale",
            "90000",
            "-movflags",
            "+faststart",
            str(dst),
        ]

    # One GPU decoder for every input, or the CPU: the probed one fits the GPU found here, and
    # only 8-bit H.264 (see decoder_for) is decoded on a VAAPI GPU.
    decs = {decoder_for(remap_source(inp)) for inp in src.inputs}
    hw: list[HwDecoder | None] = [decs.pop() if len(decs) == 1 else None]
    prog = _Progress(progress, dur, out.name) if progress is not None else None

    def run_one(enc: VideoEncoder, dec: HwDecoder | None, dst: Path) -> str:
        if prog is None:
            return run(args_for(enc, dec, dst), cwd=work)
        with prog.attempt() as pfile:
            return run(args_for(enc, dec, dst, pfile), cwd=work)

    def encode(enc: VideoEncoder, dst: Path) -> str:
        """Render into ``dst`` with ``enc``: GPU decoding first, then CPU decoding."""
        if hw[0] is not None:
            try:
                return run_one(enc, hw[0], dst)
            except FFmpegError as err:
                logger.warning(
                    "  %s: %s render failed, retrying on the CPU (%s)",
                    out.name,
                    hw[0].kind,
                    str(err).splitlines()[-1][:200],
                )
                hw[0] = None  # the CPU-encoder retry decodes on the CPU too
        return run_one(enc, None, dst)

    t0 = time.perf_counter()
    # A failed auto GPU encode is redone with libx264; out only appears once complete.
    with atomic_output(out) as part:
        text = encode_with_fallback(lambda enc: encode(enc, part), out.name)
        got = probe(part).duration
    el = time.perf_counter() - t0
    found = [m.strip() for m in READ_ERROR_RE.findall(text)]
    read_err = None
    if found or got < dur - 0.3:
        read_err = {
            "failure_time": round(start + got, 3),
            "messages": found[:5],
            "end": round(start + got, 3),
        }
        logger.warning("READ ERROR rendering %s: got %.2f of %.2f s", out.name, got, dur)
    if prog is not None:
        prog.done()
    return {
        "mode": src.mode,
        "inputs": list(src.inputs),
        "levelled": src.levelled,
        "flowstate": source_flowstate(src.mode, src.inputs),
        "preview": src.preview,
        "note": src.note,
        "seconds": round(el, 1),
        "realtime": round(dur / max(el, 1e-6), 2),
        "commands": len(lines),
        "cmd_every": cmd_every,
        "cmd_file": cmd_file.name if cmd_file is not None else None,
        "read_error": read_err,
    }


def backup_path_file(f: Path) -> Path:
    """Copy a director path file (it may be hand-edited) to the first free backup name
    (``<file>.bak``, then ``<file>.1.bak``, ``<file>.2.bak``, ...) and return it; earlier
    backups are never overwritten."""
    bak = free_backup_name(f, f.name, ".bak")
    write_atomic(bak, lambda tmp: shutil.copy2(f, tmp))
    return bak


def write_atomic(path: Path, fill: Callable[[Path], object]) -> None:
    """Have ``fill`` write a temp file next to ``path`` (``<name>.<random>.tmp``, which
    the director's ``*.json`` globs do not match), then rename it onto ``path``; a
    reader never sees a half-written ``path``. The temp file is removed on any failure."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        fill(tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def free_backup_name(f: Path, base: str, ext: str) -> Path:
    """The first free backup name for ``f`` in its folder: ``<base><ext>``, then
    ``<base>.1<ext>``, ``<base>.2<ext>``, ...

    Raises:
        OSError: If a name cannot be checked (permission denied, stale mount, name
            too long); a name that cannot be checked is never used.
    """
    bak = f.with_name(base + ext)
    n = 0
    while True:
        try:
            if not bak.exists():
                return bak
        except OSError as err:
            raise OSError(
                err.errno,
                f"Cannot check backup name, not writing a backup ({err.strerror})",
                str(bak),
            ) from err
        n += 1
        bak = f.with_name(f"{base}.{n}{ext}")


# --------------------------------------------------------------------- cutpoint clips / extend
class PanoExporter:
    """Renders the X3 moments of a ``cutpoint clips`` / ``cutpoint extend`` run.

    Args:
        project: Project folder with ``pano.json``.
        out_dir: Clips folder being written (path files go to
            ``<project>/pano_paths/<folder name>/``).
        vertical: 9:16 output.
        source_mode: ``auto`` | ``sdk`` | ``studio`` | ``dfisheye``.
        cmd_every: Frames between ``v360`` updates (default ``pano.cmd_every`` or 1).
    """

    def __init__(
        self,
        project: Path,
        out_dir: Path,
        vertical: bool = False,
        source_mode: str = "auto",
        cmd_every: int | None = None,
    ) -> None:
        self.project = project
        self.clips = load(project)
        self.fmt = Format360(1080, 1920) if vertical else Format360()
        self.vertical = vertical
        self.source_mode = source_mode
        self.cmd_every = cmd_every
        self.paths_dir = project / PATHS_DIR / out_dir.name
        self.overrides = D.config_overrides() or None  # pano.director, as `360 director`
        self._warned = False

    def handles(self, source: str | Path) -> bool:
        """True when ``source`` is a scanned X3 clip of this project."""
        return clip_for(self.clips, source) is not None

    def path_for(
        self,
        clip: X3Clip,
        name: str,
        start: float,
        end: float,
        kind: str | None,
        reuse: Path | None = None,
        selected: tuple[float, float] | None = None,
    ) -> tuple[dict[str, Any], Path]:
        """Load ``reuse`` (an earlier, possibly hand-edited path) if it covers the range
        and is not older than :data:`director.MIN_REUSE_VERSION`, else take the path
        ``360 analyze`` planned for exactly this span and these settings
        (:func:`cutpoint.pano.candidates.cached_path`; an INFO line names why it is not
        reused), else run the director
        (with the ``pano.director`` overrides of config.json, like ``cutpoint 360
        director``); the path is saved under ``pano_paths``. A re-planned ``reuse``
        file is overwritten after a copy to the first free backup name (``<file>.bak``,
        then ``<file>.1.bak``, ``<file>.2.bak``, ...); earlier backups are never
        overwritten.

        ``selected``: the moment picked for the clip, before tails, padding and the speech
        pass (source s), saved as the path's ``selected`` ``{"start", "end"}`` (``segment``
        stays the rendered span); without it, a re-planned ``reuse`` keeps its ``selected``.

        On failure ``paths_dir`` (and ``pano_paths``) is removed again if this call
        created it and it is still empty (see :func:`~cutpoint.ffmpeg_utils.output_dir`)."""
        with output_dir(self.paths_dir):
            return self._path_for(clip, name, start, end, kind, reuse, selected)

    def _path_for(
        self,
        clip: X3Clip,
        name: str,
        start: float,
        end: float,
        kind: str | None,
        reuse: Path | None,
        selected: tuple[float, float] | None = None,
    ) -> tuple[dict[str, Any], Path]:
        """:meth:`path_for` inside the existing ``paths_dir``."""
        out = self.paths_dir / (Path(name).stem + ".json")
        sel = None
        if selected is not None:
            sel = {"start": round(float(selected[0]), 3), "end": round(float(selected[1]), 3)}
        if reuse is not None and reuse.is_file():
            path = cache.load_json(reuse) or {}
            sel = sel or path.get("selected")
            pl = path.get("planned", {})
            axis = G.cached_forward_axis(Path(clip.source))  # None: no track cached, no check
            if path.get("version", 1) < D.MIN_REUSE_VERSION:
                logger.warning(
                    "  %s: version %s director path predates the current framing; re-directing",
                    reuse,
                    path.get("version", 1),
                )
            elif axis is not None and path.get("forward_axis", G.axis_name(G.FORWARD_V)) != axis:
                logger.warning(
                    "  %s: planned for another forward axis (upright clip); re-directing", reuse
                )
            elif pl.get("start", 1e9) <= start + 0.5 and pl.get("end", -1e9) >= end - 0.5:
                self._check_reused(path, reuse, kind)
                changed = sel is not None and path.get("selected") != sel
                if changed:
                    path["selected"] = sel
                if changed or reuse.resolve() != out.resolve():
                    cache.save_json(out, path, indent=1)
                return path, out
            else:
                logger.info(
                    "  %s: path %s does not cover %.1f-%.1f; re-directing",
                    name,
                    reuse.name,
                    start,
                    end,
                )
        from cutpoint.pano.candidates import plan_path  # noqa: PLC0415

        path, cached, why = plan_path(
            self.project, clip, start, end, kind, self.overrides, self.vertical
        )
        if sel is not None:
            path["selected"] = sel
        if cached:
            logger.info("  %s: director path planned by 360 analyze", name)
        elif why is not None:
            logger.info("  %s: director path re-planned (%s)", name, why)
        if reuse is not None and reuse.is_file() and reuse.resolve() == out.resolve():
            bak = backup_path_file(out)
            logger.warning("  %s: re-planned; the previous path is kept as %s", name, bak)
        cache.save_json(out, path, indent=1)
        return path, out

    def _check_reused(self, path: dict[str, Any], file: Path, kind: str | None) -> None:
        """Warn when a reused path is older than :data:`director.PATH_VERSION` or was
        planned with other director settings than the current ones (it is still used:
        it may be hand-edited)."""
        version = path.get("version", 1)
        if version < D.PATH_VERSION:
            logger.warning(
                "  %s: version %s director path (current %d); reusing it as is, delete it to "
                "re-direct",
                file,
                version,
                D.PATH_VERSION,
            )
        want = asdict(D.with_params(D.params_for(kind), self.overrides or {}))
        have = path.get("params") or {}
        diff = sorted(k for k, v in want.items() if have.get(k) != v)
        if diff:
            logger.warning(
                "  %s: planned with other director params (%s); reusing it as is, delete it "
                "to re-direct",
                file,
                ", ".join(diff[:8]) + (", ..." if len(diff) > 8 else ""),
            )

    def _modes(self, mode: str | None) -> list[str]:
        """Source modes :meth:`_source` tries, in order: ``source_mode`` alone, or a
        recorded ``mode`` that differs from it, then ``auto``."""
        if mode is None or mode == self.source_mode:
            return [self.source_mode]
        return [mode, "auto"]

    def _source(self, clip: X3Clip, mode: str | None) -> RenderSource:
        """Render source for ``mode`` (a clip's recorded mode), else ``source_mode``.

        A recorded mode that is no longer available falls back to ``auto``.
        """
        first, *fallback = self._modes(mode)
        if not fallback:
            return resolve_source(clip, first)
        try:
            src = resolve_source(clip, first)
        except (FileNotFoundError, ValueError) as err:
            logger.warning(
                "  clip %s: recorded 360 source %r unavailable (%s); using auto",
                clip.clip_id,
                mode,
                err,
            )
            return resolve_source(clip, fallback[0])
        if src.preview:
            best = resolve_source(clip, "auto")
            if not best.preview:
                logger.warning(
                    "  clip %s: re-rendering from the recorded %s source; a "
                    "better one exists now (%s), pass --pano-source auto",
                    clip.clip_id,
                    mode,
                    best.mode,
                )
        return src

    def inputs(self, source: str, mode: str | None = None) -> list[Path]:
        """Files a render of ``source`` may read, kept safe from ``clips --force``: the
        camera files (``.lrv``, ``.insv``), ``clip.source`` and the render input that
        :meth:`export` with ``source_mode=mode`` picks (falling back to ``auto``), each as
        recorded and through ``source_remap``. Never raises for an unavailable mode (the
        render reports it); empty if ``source`` is not an X3 clip of the project."""
        from cutpoint.config import remap_source  # noqa: PLC0415

        clip = clip_for(self.clips, source)
        if clip is None:
            return []
        files = [p for p in (clip.lrv, *clip.insv, clip.source) if p]
        for m in dict.fromkeys([*self._modes(mode), "auto"]):
            try:
                files += resolve_source(clip, m).inputs
                break
            except (FileNotFoundError, ValueError):
                continue
        return [q for f in files for q in (Path(f), remap_source(f))]

    def export(
        self,
        source: str,
        start: float,
        end: float,
        out: Path,
        *,
        audio_fade: float,
        kind: str | None = None,
        reuse_path: Path | None = None,
        source_mode: str | None = None,
        grade: Grade | None = None,
        selected: tuple[float, float] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Render one clip. Returns ``(read_error record or None, manifest pano record)``.

        ``selected``: the clip's picked moment, recorded in its path file (:meth:`path_for`).

        ``source_mode`` (the mode recorded for this clip by an earlier run)
        replaces the exporter's ``source_mode``; if it is no longer available
        the best available source (``auto``) is used. ``grade``: the clip's grade (``--lut``).
        """
        clip = clip_for(self.clips, source)
        if clip is None:
            raise ValueError(f"{source} is not an X3 clip of {self.project}")
        src = self._source(clip, source_mode)
        if src.preview and not self._warned:
            logger.warning(
                "360: rendering from the raw dual fisheye (ffmpeg v360): PREVIEW "
                "quality, hard lens seam along the horizon. Use `cutpoint 360 stitch` "
                "(SDK) or Insta360 Studio equirect exports for final quality."
            )
            self._warned = True
        path, pfile = self.path_for(clip, out.name, start, end, kind, reuse_path, selected)
        rec = render_segment(
            clip,
            path,
            start,
            end,
            out,
            fmt=self.fmt,
            src=src,
            audio_fade=audio_fade,
            cmd_every=self.cmd_every,
            work_dir=self.paths_dir,
            grade=grade,
        )
        logger.info(
            "  %s: 360 %s, %s; rendered %.1fs at %.2fx realtime",
            out.name,
            src.mode,
            D.describe(path),
            rec["seconds"],
            rec["realtime"],
        )
        read_err = rec.pop("read_error")
        rec.update(
            {
                "clip": clip.clip_id,
                "path_file": str(pfile),
                "format": [self.fmt.width, self.fmt.height],
            }
        )
        return read_err, rec


MIXED_SOURCES = "pano_mixed_sources"  # --json warning code of :func:`source_mix_warning`


def source_mix_warning(records: list[tuple[str, dict[str, Any]]]) -> str | None:
    """The WARNING message for the 360 clips of one run, ``(file, manifest pano record)``
    each, when their renders are not like-for-like: stitched (``sdk``, ``studio``) and dual
    fisheye sources mixed, or an equirect rendered without our IMU levelling (``levelled``:
    FlowState, no sidecar or a Studio export). None when every clip renders the same way."""
    stitched = [f for f, r in records if r.get("mode") in ("sdk", "studio")]
    fisheye = [f for f, r in records if r.get("mode") == "dfisheye"]
    levelled = [f for f, r in records if r.get("levelled")]
    parts = []
    if stitched and fisheye:
        parts.append(
            f"stitched sources ({', '.join(stitched)}) mixed with the dual fisheye "
            f"({', '.join(fisheye)})"
        )
    if levelled:
        parts.append(
            f"equirects rendered without our IMU levelling (FlowState, no sidecar or a Studio "
            f"export: {', '.join(levelled)})"
        )
    if not parts:
        return None
    return (
        f"360 clips not rendered like-for-like: {'; '.join(parts)}. Their 360 qa does not "
        "compare like-for-like with each other or with the director previews (the .lrv dual "
        "fisheye); export with --pano-source dfisheye for a like-for-like comparison"
    )


def exporter_for(
    project: Path,
    out_dir: Path,
    vertical: bool,
    source_mode: str = "auto",
    cmd_every: int | None = None,
) -> PanoExporter | None:
    """A :class:`PanoExporter` when the project has ``pano.json``, else None."""
    if not (project / "pano.json").is_file():
        return None
    return PanoExporter(project, out_dir, vertical, source_mode, cmd_every)
