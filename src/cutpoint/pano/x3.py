"""Insta360 X3 clip discovery, pairing and trailer metadata.

File naming (one recording = one *clip*, numbered ``NNN``)::

    VID_YYYYMMDD_HHMMSS_00_NNN.insv   lens 1 (5.7K mode: 2880x2880; 4K mode: both
                                      lenses side by side, 3840x1920); holds the
                                      Insta360 trailer (IMU, exposure, calibration)
    VID_YYYYMMDD_HHMMSS_10_NNN.insv   lens 2 (5.7K mode only; no trailer)
    LRV_YYYYMMDD_HHMMSS_11_NNN.lrv    1024x512 dual-fisheye proxy (5.7K mode)
    LRV_YYYYMMDD_HHMMSS_01_NNN.lrv    1024x512 dual-fisheye proxy (4K mode)

The ``.lrv`` is frame-exact with both ``.insv`` files (``_00_`` has one extra
trailing frame), and its left half is the ``_00_`` lens. The toolkit uses the
``.lrv`` as the *source identity* of a clip (analysis audio, transcript,
highlights, manifests); the ``.insv`` files are only read for the IMU (trailer,
cheap) and for rendering.

The camera's clip number repeats (a formatted card, a second camera, merged days). A
clip's id is its number, or ``YYYYMMDD_HHMMSS_NNN`` when another recording of the scan or
project has that number (:func:`assign_ids`).

On the user's mount the ``_00_`` lens looks at the sky and ``_10_`` down at the
rider, so the lens seam lies on the horizon (see ``docs/insta360-sdk.md``).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import cache
from cutpoint.insta360 import parse_name

logger = logging.getLogger(__name__)

META_VERSION = 1


@dataclass
class X3Clip:
    """One X3 recording and the files that belong to it.

    Attributes:
        clip_id: The clip number (``"035"``), or ``YYYYMMDD_HHMMSS_NNN`` when another
            recording has that number (:func:`assign_ids`).
        start_local: Wall-clock start from the file name (camera clock, local).
        mode: ``"5.7k"`` (two 2880x2880 lens files) or ``"4k"`` (one 3840x1920 file).
        lrv: The proxy (source identity for analysis and clips), or None.
        insv: ``[_00_, _10_]`` in 5.7K mode, ``[_00_]`` in 4K mode.
        width, height: Size of each ``.insv`` video stream.
        fps: Frame rate (30000/1001 on the X3).
        frames: Frame count of the proxy (= ``_10_``; ``_00_`` has one more).
        duration: Proxy duration (s).
        bytes_insv: Total ``.insv`` size.
    """

    clip_id: str
    start_local: str
    mode: str
    lrv: str | None
    insv: list[str]
    width: int = 0
    height: int = 0
    fps: float = 30000 / 1001
    frames: int = 0
    duration: float = 0.0
    bytes_insv: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def number(self) -> str:
        """The camera's three-digit clip number (``"035"``): not unique across recordings."""
        f = parse_name(self.insv[0]) if self.insv else None
        return f["num"] if f else self.clip_id[-3:]

    @property
    def recording_id(self) -> str:
        """``YYYYMMDD_HHMMSS_NNN``: unique per recording (the clip id when not parseable)."""
        f = parse_name(self.insv[0]) if self.insv else None
        return clip_key(f["date"], f["time"], f["num"]) if f else self.clip_id

    @property
    def source(self) -> str:
        """Path used as the clip's source identity (the ``.lrv``, else ``_00_``)."""
        return self.lrv or self.insv[0]

    def to_json(self) -> dict[str, Any]:
        """JSON form for ``pano.json``."""
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> X3Clip:
        """Inverse of :meth:`to_json`."""
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def probe_video_stream(path: Path) -> dict[str, Any]:
    """First video stream's width, height, nb_frames, avg_frame_rate and duration.

    ffprobe reads the staged copy if there is one; ``{}`` if it fails.
    """
    from cutpoint.errors import FFmpegError  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import ffprobe_json  # noqa: PLC0415

    try:
        data = ffprobe_json(
            path,
            streams="v:0",
            entries="stream=width,height,nb_frames,avg_frame_rate,duration",
            timeout=120,
        )
    except (OSError, FFmpegError) as err:
        logger.debug("ffprobe %s failed: %s", path, err)
        return {}
    return (data.get("streams") or [{}])[0]


def _rate(s: str | None) -> float:
    try:
        a, b = (s or "0/1").split("/")
        return float(a) / float(b) if float(b) else 0.0
    except ValueError:
        return 0.0


def read_trailer_meta(insv00: Path) -> dict[str, Any]:
    """Camera, firmware and factory lens calibration from the ``_00_`` trailer.

    ``offset`` is Insta360's calibration string as numbers: lens count, then
    per lens (image-circle radius?, cx, cy, yaw, pitch, roll) in the
    ``sensor`` space (11904x5952 on the X3), then the sensor size. The exact
    model is undocumented (research doc section 1.2); it is recorded for a
    future calibrated stitch and not used by the ffmpeg renderer. The camera's serial
    number is not read: project folders get shared.
    """
    import telemetry_parser as tp  # noqa: PLC0415

    parser = tp.Parser(str(insv00))
    groups = parser.telemetry()
    md = (groups[0].get("Default") or {}).get("Metadata") or {} if groups else {}
    off = md.get("offset") or []
    lenses = []
    if off and int(off[0]) == 2 and len(off) >= 15:
        for i in range(2):
            r, cx, cy, yaw, pitch, roll = off[1 + 6 * i : 7 + 6 * i]
            lenses.append(
                {"radius": r, "cx": cx, "cy": cy, "yaw": yaw, "pitch": pitch, "roll": roll}
            )
    return {
        "camera": f"{parser.camera} {parser.model}".strip(),
        "firmware": md.get("fw_version"),
        "calibration": {
            "raw": off,
            "lenses": lenses,
            "sensor": off[13:15] if len(off) >= 15 else None,
            "window_crop": md.get("window_crop_info"),
        },
        "rolling_shutter_ms": md.get("rolling_shutter_time"),
        "dimension": md.get("dimension"),
    }


def clip_key(date: str, hms: str, num: str) -> str:
    """A recording's long id: ``YYYYMMDD_HHMMSS_NNN``."""
    return f"{date}_{hms}_{num}"


def assign_ids(clips: list[X3Clip]) -> None:
    """Set the ids of ``clips`` (one project): the clip number, or ``YYYYMMDD_HHMMSS_NNN``
    when another recording among them has that number. A clip without an X3 name keeps
    its id."""
    shared = Counter(n for n, _ in {(c.number, c.recording_id) for c in clips})
    for c in clips:
        if c.insv and parse_name(c.insv[0]) is not None:
            c.clip_id = c.recording_id if shared[c.number] > 1 else c.number


def group_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """X3 files by clip id (as :func:`assign_ids`, over ``paths``): ``date``, ``time``, ``num`` and, if found,
    ``lens00``, ``lens10``, ``lrv``.

    Pure (names only); paths whose name does not match ``cutpoint.insta360.NAME_RE`` are
    ignored. A clip is usable for 360 analysis when it has both ``lens00`` and ``lrv``.
    Files pair by date, time and number, so two recordings with one number stay apart.
    """
    groups: dict[str, dict[str, Any]] = {}
    for p in paths:
        f = parse_name(p.name)
        if f is None:
            continue
        g = groups.setdefault(
            clip_key(f["date"], f["time"], f["num"]),
            {"date": f["date"], "time": f["time"], "num": f["num"]},
        )
        if f["ext"] == "lrv":
            g["lrv"] = p
        else:
            g[f"lens{f['lens']}"] = p
    shared = Counter(g["num"] for g in groups.values())
    return {k if shared[g["num"]] > 1 else g["num"]: g for k, g in sorted(groups.items())}


def duplicate_numbers(clips: Iterable[X3Clip]) -> dict[str, list[str]]:
    """Clip numbers that more than one recording has: ``{number: [recording ids]}``."""
    ids: dict[str, set[str]] = {}
    for c in clips:
        ids.setdefault(c.number, set()).add(c.recording_id)
    return {n: sorted(v) for n, v in sorted(ids.items()) if len(v) > 1}


def select(clips: list[X3Clip], wanted: Iterable[str], option: str) -> list[X3Clip]:
    """The clips of ``wanted`` (in ``clips`` order): clip ids, ``YYYYMMDD_HHMMSS_NNN`` ids,
    or clip numbers that one recording has. Unknown names select nothing.

    Raises:
        cutpoint.jsonout.ArgumentError: For a number more than one recording uses.
    """
    from cutpoint.jsonout import ArgumentError  # noqa: PLC0415

    picked: set[int] = set()
    for w in wanted:
        hits = [c for c in clips if w in (c.clip_id, c.recording_id)]
        hits = hits or [c for c in clips if c.number == w]
        ids = sorted({c.recording_id for c in hits})
        if len(ids) > 1:
            raise ArgumentError(
                f"{option} {w}: more than one recording has clip number {w}; "
                f"name one by its id: {', '.join(ids)}"
            )
        picked |= {id(c) for c in hits}
    return [c for c in clips if id(c) in picked]


def scan(folder: Path, with_meta: bool = True) -> list[X3Clip]:
    """Discover the X3 clips in ``folder`` (read-only), paired and probed.

    Args:
        folder: Folder with ``.insv`` / ``.lrv`` files (e.g. ``/media/x3/360`` or
            ``D:\\Videos\\360``).
        with_meta: Also read the trailer metadata (camera, calibration).

    Returns:
        Clips ordered by start time, with ids as :func:`assign_ids` gives over the folder.

    Raises:
        FileNotFoundError: If the folder does not exist or holds no X3 files.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    groups = group_files(sorted(folder.iterdir()))
    if not groups:
        raise FileNotFoundError(f"No Insta360 .insv/.lrv files in {folder}")
    clips = []
    for key, g in groups.items():
        insv = [g[k] for k in ("lens00", "lens10") if k in g]
        if "lens00" not in g:
            logger.warning(
                "clip %s: no _00_.insv (only %s); skipped",
                key,
                ", ".join(p.name for p in insv) or "a proxy",
            )
            continue
        start = datetime.strptime(g["date"] + g["time"], "%Y%m%d%H%M%S")
        cached = _cached_probe(g, insv)
        clip = X3Clip(
            clip_id=key,
            start_local=start.isoformat(),
            mode="5.7k" if "lens10" in g else "4k",
            lrv=str(g["lrv"]) if "lrv" in g else None,
            insv=[str(p) for p in insv],
            **cached,
        )
        if with_meta:
            mpath = cache.source_dir(Path(clip.source)) / f"x3-meta-v{META_VERSION}.json"
            meta = cache.load_json(mpath)
            # a failed read is never cached (an older file may hold one): retry next run
            if not isinstance(meta, dict) or "error" in meta:
                try:
                    meta = read_trailer_meta(insv[0])
                except Exception as err:  # noqa: BLE001 - Rust parser raises plain Exception
                    logger.warning("clip %s: trailer unreadable (%s)", key, err)
                    meta = {"error": str(err)}
                else:
                    cache.save_json(mpath, meta)
            clip.meta = meta
        clips.append(clip)
    return clips


def _cached_probe(g: dict[str, Any], insv: list[Path]) -> dict[str, Any]:
    """Width/height/frames/duration (ffprobe), cached per clip."""
    key_path = g.get("lrv") or insv[0]
    path = cache.source_dir(key_path) / f"x3-probe-v{META_VERSION}.json"
    hit = cache.load_json(path)
    if hit is not None:
        return hit
    t0 = time.perf_counter()
    vs = probe_video_stream(insv[0])
    ps = probe_video_stream(g["lrv"]) if "lrv" in g else vs
    out = {
        "width": int(vs.get("width") or 0),
        "height": int(vs.get("height") or 0),
        "fps": _rate(ps.get("avg_frame_rate")) or 30000 / 1001,
        "frames": int(ps.get("nb_frames") or 0),
        "duration": float(ps.get("duration") or 0.0),
        "bytes_insv": sum(p.stat().st_size for p in insv),
    }
    if not vs or not ps:
        return out  # ffprobe failed: do not cache zeros, retry next run
    cache.save_json(path, out)
    logger.debug("probe %s in %.1fs", key_path, time.perf_counter() - t0)
    return out
