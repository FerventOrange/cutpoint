"""Horizon roll of a rendered 360 clip, measured from the ``v360`` commands it was rendered with.

The renderer (:func:`cutpoint.pano.render.render_segment`) writes a ``.cmd`` file that
``sendcmd`` feeds to ``v360``: a header comment with the initial rotation, then
incremental ``roll`` / ``yaw`` / ``pitch`` commands. This module replays that file the way
ffmpeg applies it and checks where the true up (the camera's IMU gravity) lands in the
output picture. It does not use the renderer's stabilisation model, the director path or
the renderer's rotation helpers, so a bug in either (wrong levelling, a sign or order
error in the deltas, a wrong initial rotation, commands timed for the wrong frame) shows
up as roll. What it shares with the renderer is the IMU orientation track and the
measured equirect calibration :data:`~cutpoint.pano.geometry.SDK_EQUIRECT_FROM_CAMERA`.

``v360`` conventions (ffmpeg 9.0.2, see :mod:`cutpoint.pano.geometry`): an output ray
``d`` samples the input along ``R @ d``; ``yaw`` turns about +y (+z toward +x), ``pitch``
about +x (+z toward -y) and ``roll`` about +z (+x toward +y), all right-handed. The
initial options with ``rorder=ryp`` give ``R = roll @ yaw @ pitch``; each runtime command
right-multiplies the current ``R`` by its single-axis turn, in the order listed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cutpoint.pano import geometry as G
from cutpoint.pano.render import CMD_HEADER

AXES = {"yaw": (0.0, 1.0, 0.0), "pitch": (1.0, 0.0, 0.0), "roll": (0.0, 0.0, 1.0)}


@dataclass(frozen=True)
class Commands:
    """A parsed ``.cmd`` file.

    Attributes:
        projection: ``v360`` input, ``dfisheye`` or ``equirect``.
        start: Source time (s) of the seek; command times are relative to it.
        fps: Source frame rate.
        init: Initial rotation ``(3, 3)``.
        times: Command times ``(N,)`` (relative s, increasing).
        after: Rotation in effect after each command line ``(N, 3, 3)``.
    """

    projection: str
    start: float
    fps: float
    init: np.ndarray
    times: np.ndarray
    after: np.ndarray

    def rotation(self, rel: np.ndarray) -> np.ndarray:
        """``v360`` rotation ``(K, 3, 3)`` in effect at relative times ``rel``."""
        i = np.searchsorted(self.times, np.asarray(rel, float), side="right") - 1
        stack = np.concatenate([self.init[None], self.after])
        return stack[i + 1]


def turn(name: str, deg: float) -> np.ndarray:
    """Right-handed turn by ``deg`` about ``v360``'s axis for ``name`` (Rodrigues)."""
    x, y, z = AXES[name]
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    a = math.radians(deg)
    return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)


def read_commands(path: Path) -> Commands | None:
    """Parse a renderer ``.cmd`` file, or None when it has no :data:`CMD_HEADER` line
    (rendered before the header existed) or cannot be read.

    Raises:
        ValueError: If a command line is malformed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(CMD_HEADER + " "):
        return None
    head = dict(kv.split("=", 1) for kv in lines[0][len(CMD_HEADER) :].split())
    if head.get("rorder") != "ryp":
        raise ValueError(f"{path}: unsupported rorder {head.get('rorder')!r}")
    cur = turn("roll", float(head["roll"])) @ turn("yaw", float(head["yaw"]))
    cur = cur @ turn("pitch", float(head["pitch"]))
    init = cur
    times, after = [], []
    for ln in lines[1:]:
        if ln.startswith("#"):
            continue
        t, _, body = ln.partition(" ")
        for part in body.rstrip(";").split(","):
            target, name, val = part.split()
            if not target.startswith("v360") or name not in AXES:
                continue  # h_fov / v_fov
            cur = cur @ turn(name, float(val))
        times.append(float(t))
        after.append(cur)
    return Commands(
        projection=head["input"],
        start=float(head["start"]),
        fps=float(head["fps"]),
        init=init,
        times=np.array(times, float),
        after=np.array(after, float).reshape(-1, 3, 3),
    )


def emitted_roll(cmd: Commands, orient: G.Orientation, src_times: np.ndarray) -> np.ndarray:
    """Roll (deg) of the true horizon in the rendered picture at source ``src_times``.

    Args:
        cmd: The clip's parsed commands.
        orient: The clip's IMU orientation track (camera -> world per source frame,
            world -y up).
        src_times: Source times (s).

    Each time is shown as source frame ``round(t * fps)``, but not before the first
    frame after the seek (a clip's recorded start can lie up to a frame before it); the
    commands in effect are those timed up to that frame's relative timestamp (the
    renderer sends them just before it). Up in output coordinates is ``u = R.T @ up_in``
    (``up_in`` in the ``v360`` input frame), roll = atan2(u_x, -u_y).
    """
    t = np.asarray(src_times, float)
    first = round(cmd.start * cmd.fps)  # the header start is frame-aligned
    j = np.clip(np.round(t * cmd.fps).astype(int), first, len(orient.q) - 1)
    r = cmd.rotation(j / cmd.fps - cmd.start + 1e-6)
    up = np.einsum("kji,j->ki", orient.cam_to_world(j), [0.0, -1.0, 0.0])  # world up, camera axes
    if cmd.projection == "equirect":
        up = up @ G.SDK_EQUIRECT_FROM_CAMERA.T
    u = np.einsum("kji,kj->ki", r, up)  # R.T @ up
    return np.degrees(np.arctan2(u[:, 0], -u[:, 1]))
