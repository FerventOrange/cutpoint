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

On the user's mount the ``_00_`` lens looks at the sky and ``_10_`` down at the
skier, so the lens seam lies on the horizon (see ``docs/insta360-sdk.md``).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import cache

logger = logging.getLogger(__name__)

NAME_RE = re.compile(
    r"^(?P<kind>VID|LRV)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<lens>\d{2})_"
    r"(?P<num>\d{3})\.(?P<ext>insv|lrv)$",
    re.IGNORECASE,
)
META_VERSION = 1


def parse_name(path: str | Path) -> dict[str, str] | None:
    """Fields of an X3 file name (``kind``, ``date``, ``time``, ``lens``, ``num``, ``ext``)."""
    m = NAME_RE.match(Path(path).name)
    return (
        {
            k: v.upper() if k in ("kind",) else v.lower() if k == "ext" else v
            for k, v in m.groupdict().items()
        }
        if m
        else None
    )


def imu_source(path: str | Path) -> Path | None:
    """The ``_00_.insv`` that carries the IMU trailer for any file of a clip.

    telemetry-parser rejects ``.lrv`` files and the ``_10_`` lens has no
    trailer, so every X3 telemetry read goes through the ``_00_`` file. It
    only reads the trailer at the end of the file (a 7.8 GB clip parses in
    ~2 s even over a network share), so no staging is needed.

    Returns:
        The path (which exists), or None for a non-X3 file or a missing ``_00_``.
    """
    f = parse_name(path)
    if f is None:
        return None
    cand = Path(path).with_name(f"VID_{f['date']}_{f['time']}_00_{f['num']}.insv")
    if cand.exists():
        return cand
    # Match names case-insensitively, so .INSV / vid_ also work on case-sensitive file systems.
    try:
        siblings = list(Path(path).parent.iterdir())
    except OSError:
        return None
    exact = [p for p in siblings if p.name.casefold() == cand.name.casefold()]
    if exact:
        return exact[0]
    loose = re.compile(rf"VID_.*_00_{f['num']}\.insv", re.IGNORECASE)
    hits = [p for p in siblings if loose.fullmatch(p.name)]
    return hits[0] if len(hits) == 1 else None


@dataclass
class X3Clip:
    """One X3 recording and the files that belong to it.

    Attributes:
        clip_id: The three-digit clip number (``"035"``).
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
    from cutpoint.ffmpeg_utils import FFmpegError, ffprobe_json  # noqa: PLC0415

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
    future calibrated stitch and not used by the ffmpeg renderer.
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
        "serial": md.get("serial_number"),
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


def scan(folder: Path, with_meta: bool = True) -> list[X3Clip]:
    """Discover the X3 clips in ``folder`` (read-only), paired and probed.

    Args:
        folder: Folder with ``.insv`` / ``.lrv`` files (e.g. ``/media/x3/360`` or
            ``D:\\Videos\\360``).
        with_meta: Also read the trailer metadata (camera, calibration).

    Returns:
        Clips ordered by clip number.

    Raises:
        FileNotFoundError: If the folder does not exist or holds no X3 files.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    groups: dict[str, dict[str, Any]] = {}
    for p in sorted(folder.iterdir()):
        f = parse_name(p.name)
        if f is None:
            continue
        g = groups.setdefault(f["num"], {"date": f["date"], "time": f["time"]})
        if f["ext"] == "lrv":
            g["lrv"] = p
        else:
            g[f"lens{f['lens']}"] = p
    if not groups:
        raise FileNotFoundError(f"No Insta360 .insv/.lrv files in {folder}")
    clips = []
    for num in sorted(groups):
        g = groups[num]
        insv = [g[k] for k in ("lens00", "lens10") if k in g]
        if "lens00" not in g:
            logger.warning(
                "clip %s: no _00_.insv (only %s); skipped",
                num,
                ", ".join(p.name for p in insv) or "a proxy",
            )
            continue
        start = datetime.strptime(g["date"] + g["time"], "%Y%m%d%H%M%S")
        cached = _cached_probe(g, insv)
        clip = X3Clip(
            clip_id=num,
            start_local=start.isoformat(),
            mode="5.7k" if "lens10" in g else "4k",
            lrv=str(g["lrv"]) if "lrv" in g else None,
            insv=[str(p) for p in insv],
            **cached,
        )
        if with_meta:
            mpath = cache.source_dir(Path(clip.source)) / f"x3-meta-v{META_VERSION}.json"
            meta = cache.load_json(mpath)
            if meta is None:
                try:
                    meta = read_trailer_meta(insv[0])
                except Exception as err:  # noqa: BLE001 - Rust parser raises plain Exception
                    logger.warning("clip %s: trailer unreadable (%s)", num, err)
                    meta = {"error": str(err)}
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
