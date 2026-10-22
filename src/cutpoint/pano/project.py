"""``<project>/pano.json``, the ``pano`` section of ``config.json`` and render sources.

``config.json`` (all keys optional; on Windows write paths such as
``"C:\\\\Tools\\\\MediaSDK\\\\bin\\\\MediaSDKTest.exe"``)::

    "pano": {
      "sdk_exe": "/opt/MediaSDK-<version>-linux/bin/MediaSDKTest",
      "stitched_dir": "/data/footage/360-stitched",
      "studio_dirs": ["/data/footage/360-studio"],
      "stitch_args": ["-stitch_type", "optflow", ...],
      "stitch_timeout_s": 7200,
      "cmd_every": 1,
      "level_strength": 0.7,
      "tilt_follow_s": 0.5,
      "imu_offset_s": 0.0,
      "equirect_yaw_offset": 0.0,
      "default_style": "ski"
    }

``stitched_dir`` and ``studio_dirs`` default to ``<CUTPOINT_HOME>/360-stitched``
and ``<CUTPOINT_HOME>/360-studio``; ``default_style`` is the ``--style`` of
``cutpoint 360 analyze`` when none is given (default ``ski``).

``CUTPOINT_INSTA360_SDK`` (path of ``MediaSDKTest``, ``.exe`` on Windows) overrides ``sdk_exe``.
When neither is set, the SDK is auto-detected under
``<CUTPOINT_HOME>/sdk/insta360/`` (any depth; on Linux also
``/opt/MediaSDK-<version>-linux/bin/``), newest version first.
``stitch_timeout_s`` (default: none) kills a clip's stitch after that many seconds.

Render sources, best first (``source_mode``):

``sdk``       ``<stitched_dir>/<VID stem>_NNN.equirect.mp4`` written by
              ``cutpoint 360 stitch`` (Insta360 Media SDK, optical flow, no
              FlowState), levelled with our IMU track like ``dfisheye``. Its
              sidecar ``<stem>.equirect.json`` records the stitch arguments; a
              stitch with ``-enable_flowstate`` or without a sidecar (older
              runs) is used as is, with a warning: its yaw 0 does not follow
              the director's display frame, so restitch it (``--force``)
``studio``    an equirect (2:1) MP4/MOV exported from Insta360 Studio into one
              of ``studio_dirs`` (default: those folders plus ``_stitched`` and
              ``export`` next to the camera files), matched to the clip by the
              ``VID_YYYYMMDD_HHMMSS`` stem or ``_NNN`` clip number in its name,
              and full length (a trimmed export loses the time mapping);
              assumed FlowState and used as is, with the same warning
``dfisheye``  the ``.insv`` dual fisheye straight into ffmpeg ``v360``
              (always available; preview quality: hard seam on the horizon)
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import cache, paths, state
from cutpoint.config import path_key, remap_source, source_path
from cutpoint.pano.x3 import X3Clip, parse_name

logger = logging.getLogger(__name__)

PANO_FILE = "pano.json"
DEFAULT_STYLE = "ski"
DEFAULT_STITCH_ARGS = (
    "-stitch_type",
    "optflow",
    "-output_size",
    "5760x2880",
    "-enable_h265_encoder",
    "-bitrate",
    "100000000",
    "-enable_stitchfusion",
)
FLOWSTATE_ARG = "-enable_flowstate"
EQUIRECT_META_VERSION = 1
MODES = ("sdk", "studio", "dfisheye")
STITCH_TYPES = ("optflow", "dynamicstitch", "aistitch")
SDK_EXE_NAME = "MediaSDKTest.exe" if sys.platform == "win32" else "MediaSDKTest"
_OUTPUT_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")
_SDK_VERSION_RE = re.compile(r"MediaSDK-(\d+(?:\.\d+)+)", re.IGNORECASE)


def pano_config() -> dict[str, Any]:
    """The ``pano`` section of ``config.json`` (``{}`` when absent).

    Raises:
        ValueError: If ``pano`` is set but is not an object.
    """
    from cutpoint.config import load_config_data  # noqa: PLC0415

    section = load_config_data().get("pano") or {}
    if not isinstance(section, dict):
        raise ValueError(f"{paths.config_file()}: 'pano' must be an object")
    return section


def default_stitched_dir() -> Path:
    """``<CUTPOINT_HOME>/360-stitched``."""
    return paths.home() / "360-stitched"


def default_studio_dirs() -> tuple[Path, ...]:
    """``(<CUTPOINT_HOME>/360-studio,)``."""
    return (paths.home() / "360-studio",)


def default_style() -> str:
    """``--style`` of ``cutpoint 360 analyze`` when none is given (``pano.default_style``)."""
    return str(pano_config().get("default_style") or DEFAULT_STYLE)


def stitched_dir() -> Path:
    """Where SDK-stitched equirects go (``pano.stitched_dir``)."""
    configured = pano_config().get("stitched_dir")
    return Path(configured) if configured else default_stitched_dir()


def sdk_version(exe: str | Path) -> str | None:
    """The version in the ``MediaSDK-<version>-...`` folder of ``exe``'s path, or None."""
    for part in reversed(Path(exe).parts):
        m = _SDK_VERSION_RE.search(part)
        if m:
            return m.group(1)
    return None


def _version_key(exe: Path) -> tuple[tuple[int, ...], str]:
    v = sdk_version(exe)
    return (tuple(int(x) for x in v.split(".")) if v else (), str(exe))


def find_sdk(roots: list[Path] | None = None) -> Path | None:
    """Auto-detected SDK executable: the newest version found, or None.

    Searches ``<CUTPOINT_HOME>/sdk/insta360/`` recursively for
    :data:`SDK_EXE_NAME`, and on Linux also ``/opt/MediaSDK-*-linux/bin/``.

    Args:
        roots: Folders to search recursively instead of the defaults (tests).
    """
    hits: list[Path] = []
    for root in roots if roots is not None else [paths.home() / "sdk" / "insta360"]:
        if root.is_dir():
            hits += [p for p in root.rglob(SDK_EXE_NAME) if p.is_file()]
    if roots is None and sys.platform.startswith("linux"):
        hits += [
            p for p in Path("/opt").glob(f"MediaSDK-*-linux/bin/{SDK_EXE_NAME}") if p.is_file()
        ]
    return max(hits, key=_version_key) if hits else None


def sdk_location() -> tuple[Path | None, str]:
    """``(executable, origin)``: where the SDK comes from.

    ``origin`` is ``env`` (``CUTPOINT_INSTA360_SDK``),
    ``config`` (``pano.sdk_exe``), ``auto`` (:func:`find_sdk`) or ``none``. A
    configured path is returned even when it does not exist.
    """
    raw = paths.env("INSTA360_SDK")
    if raw:
        return Path(raw), "env"
    raw = pano_config().get("sdk_exe")
    if raw:
        return Path(raw), "config"
    found = find_sdk()
    return (found, "auto") if found else (None, "none")


def sdk_exe() -> Path | None:
    """The SDK executable: ``CUTPOINT_INSTA360_SDK``, else ``pano.sdk_exe``, else
    auto-detected (see :func:`sdk_location`)."""
    return sdk_location()[0]


def validate_stitch_args(args: list[str]) -> list[str]:
    """``args`` as strings, checking every ``-stitch_type`` and ``-output_size`` value.

    Cutpoint accepts only the stitch types listed in Insta360's public docs
    (:data:`STITCH_TYPES`) and only 2:1 (equirect) output sizes; anything else is
    rejected here, before the run.

    Raises:
        ValueError: If ``-stitch_type`` has no value or an unsupported one, or
            ``-output_size`` is not ``<W>x<H>`` with ``W == 2 * H``.
    """
    out = [str(a) for a in args]
    for i, a in enumerate(out):
        value = out[i + 1] if i + 1 < len(out) else None
        if a == "-stitch_type" and value not in STITCH_TYPES:
            raise ValueError(
                f"Unsupported -stitch_type {value!r} in pano.stitch_args "
                f"(use one of: {', '.join(STITCH_TYPES)})"
            )
        if a == "-output_size":
            m = _OUTPUT_SIZE_RE.match(value or "")
            if not m or int(m.group(1)) <= 0 or int(m.group(1)) != 2 * int(m.group(2)):
                raise ValueError(
                    f"Invalid -output_size {value!r} in pano.stitch_args: use <width>x<height> "
                    "with width = 2 x height (equirect 2:1), e.g. 5760x2880"
                )
    return out


def stitch_args() -> list[str]:
    """``pano.stitch_args`` (default :data:`DEFAULT_STITCH_ARGS`), validated.

    Only a JSON list is accepted, one argument per item (strings; plain numbers
    such as a ``-bitrate`` value are stringified). A single string such as
    ``"-stitch_type optflow"`` is rejected rather than split, so it can never be
    passed on character by character.

    Raises:
        ValueError: If it is not a list, an item is not a string or number, it
            has an unsupported ``-stitch_type``, or an ``-output_size`` that is
            not ``<W>x<H>`` with ``W == 2 * H`` (2:1).
    """
    raw = pano_config().get("stitch_args")
    if raw is not None and not isinstance(raw, list):
        raise ValueError(f"{paths.config_file()}: 'pano.stitch_args' must be a list")
    for item in raw or ():
        if isinstance(item, bool) or not isinstance(item, str | int | float):
            raise ValueError(
                f"{paths.config_file()}: 'pano.stitch_args' items must be strings, got {item!r}"
            )
    return validate_stitch_args(raw or list(DEFAULT_STITCH_ARGS))


def stitch_timeout_s() -> float | None:
    """Per-clip stitch timeout (``pano.stitch_timeout_s``; None = no limit).

    Raises:
        ValueError: If set to something other than a positive number.
    """
    raw = pano_config().get("stitch_timeout_s")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
        raise ValueError(
            f"{paths.config_file()}: 'pano.stitch_timeout_s' must be a positive number"
        )
    return float(raw)


# --------------------------------------------------------------------------- pano.json
def save(project: Path, clips: list[X3Clip], camera_dirs: list[Path]) -> Path:
    """Write ``<project>/pano.json`` (merging clips from earlier scans of other folders)."""
    project.mkdir(parents=True, exist_ok=True)
    path = project / PANO_FILE
    old = cache.load_json(path) or {}
    by_src = {c["lrv"] or c["insv"][0]: c for c in old.get("clips", [])}
    for c in clips:
        by_src[c.source] = c.to_json()
    dirs = list(dict.fromkeys([*old.get("camera_dirs", []), *map(str, camera_dirs)]))
    cache.save_json(
        path,
        {
            "version": 1,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "camera_dirs": dirs,
            "clips": sorted(by_src.values(), key=lambda c: (c["start_local"], c["clip_id"])),
        },
    )
    return path


def load(project: Path) -> dict[str, X3Clip]:
    """``{normalised source path: clip}`` from ``pano.json`` (``{}`` when absent)."""
    data = cache.load_json(project / PANO_FILE)
    if not data:
        return {}
    return {norm(c.source): c for c in (X3Clip.from_json(d) for d in data.get("clips", []))}


def camera_dirs(project: Path) -> list[Path]:
    """Camera folders recorded by ``cutpoint 360 scan``."""
    data = cache.load_json(project / PANO_FILE) or {}
    return [Path(p) for p in data.get("camera_dirs", [])]


def norm(path: str | Path) -> str:
    """Forward-slash path for lookups (case-folded on Windows, see :func:`path_key`)."""
    return path_key(str(path).replace("\\", "/"))


def clip_for(clips: dict[str, X3Clip], source: str | Path) -> X3Clip | None:
    """The X3 clip whose source identity is ``source`` (also by file name)."""
    hit = clips.get(norm(source))
    if hit is not None:
        return hit
    name = path_key(source_path(source).name)
    for c in clips.values():
        if path_key(source_path(c.source).name) == name:
            return c
    return None


# --------------------------------------------------------------------------- sources
@dataclass(frozen=True)
class RenderSource:
    """Video input(s) for rendering one clip.

    Attributes:
        mode: ``sdk``, ``studio`` or ``dfisheye``.
        inputs: Files fed to ffmpeg (two lens files are stacked side by side).
        projection: ``v360`` input projection: ``equirect`` or ``dfisheye``.
        levelled: True when the path angles are used on the input as is
            (FlowState SDK stitches and Studio exports, plus
            ``equirect_yaw_offset``); False when the renderer composes our IMU
            levelling (dual fisheye and plain SDK stitches).
        preview: True for the seam-y dual-fisheye mode.
        note: Human-readable origin.
    """

    mode: str
    inputs: tuple[str, ...]
    projection: str
    levelled: bool
    preview: bool
    note: str


def _studio_dirs(clip: X3Clip, extra: list[Path]) -> list[Path]:
    cfg = pano_config().get("studio_dirs")
    dirs = [Path(d) for d in (cfg if isinstance(cfg, list) else default_studio_dirs())]
    # The original camera folder first, then the remapped copy (its folders moved with it).
    cams = dict.fromkeys([Path(clip.insv[0]).parent, remap_source(clip.insv[0]).parent])
    dirs += [c / s for c in cams for s in ("_stitched", "export", "Export")] + extra
    return list(dict.fromkeys(d for d in dirs if d.is_dir()))


_NUM_RE = re.compile(r"(?:^|[_\-\s])(\d{3})(?:[_\-\s.]|$)")


def find_studio_export(
    clip: X3Clip, extra_dirs: list[Path] | None = None
) -> tuple[Path | None, str]:
    """An Insta360 Studio equirect export of ``clip``, or ``(None, reason)``."""
    from cutpoint.ffmpeg_utils import FFmpegError, probe  # noqa: PLC0415

    f = parse_name(clip.insv[0]) or {}
    stem = f"{f.get('date', '')}_{f.get('time', '')}"
    reasons = []
    for d in _studio_dirs(clip, extra_dirs or []):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".mp4", ".mov") or p.name.lower().endswith(".partial"):
                continue
            if parse_name(p) is not None:
                continue
            nums = _NUM_RE.findall(p.stem)
            if stem not in p.name and clip.clip_id not in nums:
                continue
            try:
                info = probe(p)
            except FFmpegError as err:
                reasons.append(f"{p.name}: unreadable ({err})")
                continue
            if abs(info.width - 2 * info.height) > 4:
                reasons.append(
                    f"{p.name}: {info.width}x{info.height} is not a 2:1 equirect "
                    "(a reframed flat export is ordinary footage: cutpoint analyze it)"
                )
                continue
            if clip.duration and abs(info.duration - clip.duration) > 1.0:
                reasons.append(
                    f"{p.name}: {info.duration:.1f}s vs clip {clip.duration:.1f}s "
                    "(trimmed export: no time mapping)"
                )
                continue
            return p, f"Studio export {p}"
    return None, "; ".join(reasons) or "no Studio export found"


def sdk_output(clip: X3Clip) -> Path:
    """Path of the SDK-stitched equirect of ``clip``."""
    stem = Path(clip.insv[0]).name.replace("_00_", "_").rsplit(".", 1)[0]
    return stitched_dir() / f"{stem}.equirect.mp4"


def equirect_sidecar(equirect: Path) -> Path:
    """``<stem>.equirect.json`` next to ``<stem>.equirect.mp4``: how it was stitched."""
    return equirect.with_suffix(".json")


def equirect_flowstate(equirect: Path) -> bool | None:
    """Whether the SDK stitch ``equirect`` used FlowState (per its sidecar), or None
    when the sidecar is missing or unreadable."""
    meta = cache.load_json(equirect_sidecar(equirect))
    if not isinstance(meta, dict) or not isinstance(meta.get("stitch_args"), list):
        return None
    return FLOWSTATE_ARG in meta["stitch_args"]


_flowstate_warned: set[str] = state.register_container(__name__, "_flowstate_warned", set())


def _warn_flowstate(path: Path, why: str, fix: str) -> None:
    """Warn (once per file) that an equirect is rendered without our IMU levelling."""
    if str(path) in _flowstate_warned:
        return
    _flowstate_warned.add(str(path))
    logger.warning(
        "%s: %s. Path angles are applied to it directly, but its yaw 0 does not "
        "follow the director's (the smoothed camera heading), so the framing drifts "
        "as the heading changes. To fix: %s.",
        path.name,
        why,
        fix,
    )


def resolve_source(clip: X3Clip, prefer: str = "auto") -> RenderSource:
    """Pick the render source of a clip.

    Args:
        clip: The clip.
        prefer: ``auto`` (sdk > studio > dfisheye) or one mode (an unavailable
            mode raises).

    An SDK stitch is ``levelled=False`` when its sidecar shows no FlowState;
    FlowState, sidecar-less and Studio equirects are ``levelled=True`` and log
    a warning once per file (see :func:`equirect_flowstate`).

    Raises:
        FileNotFoundError: If a requested mode has no file.
        ValueError: For an unknown mode.
    """
    if prefer not in ("auto", *MODES):
        raise ValueError(f"Unknown 360 source mode {prefer!r} (auto, {', '.join(MODES)})")
    if prefer in ("auto", "sdk"):
        p = sdk_output(clip)
        if p.is_file():
            flowstate = equirect_flowstate(p)
            if flowstate is False:
                return RenderSource(
                    "sdk", (str(p),), "equirect", False, False, f"SDK-stitched equirect {p}"
                )
            _warn_flowstate(
                p,
                ("no sidecar " + equirect_sidecar(p).name + ", assuming FlowState")
                if flowstate is None
                else "stitched with FlowState",
                f"restitch it with `cutpoint 360 stitch <project> --clips {clip.clip_id} --force`",
            )
            return RenderSource(
                "sdk", (str(p),), "equirect", True, False, f"SDK-stitched equirect {p}"
            )
        if prefer == "sdk":
            raise FileNotFoundError(
                f"clip {clip.clip_id}: no SDK-stitched file {p}; run "
                "`cutpoint 360 stitch` (needs MediaSDKTest.exe)"
            )
    if prefer in ("auto", "studio"):
        p, why = find_studio_export(clip)
        if p is not None:
            _warn_flowstate(
                p,
                "Studio export, assumed FlowState",
                "prefer `cutpoint 360 stitch` for accurate framing",
            )
            return RenderSource("studio", (str(p),), "equirect", True, False, why)
        if prefer == "studio":
            raise FileNotFoundError(f"clip {clip.clip_id}: {why}")
    return RenderSource(
        "dfisheye",
        tuple(clip.insv),
        "dfisheye",
        False,
        True,
        "ffmpeg dual fisheye from .insv (preview: hard seam on the horizon)",
    )
