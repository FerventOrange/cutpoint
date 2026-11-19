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
      "max_follow_deg": 30,
      "imu_offset_s": 0.0,
      "equirect_yaw_offset": 0.0,
      "default_style": "action-cam",
      "auto_stitch": true
    }

``stitched_dir`` and ``studio_dirs`` default to ``<CUTPOINT_HOME>/360-stitched``
and ``<CUTPOINT_HOME>/360-studio``; ``default_style`` is the ``--style`` of
``cutpoint 360 analyze`` when none is given (default ``action-cam``).
``auto_stitch`` (default ``true``): ``cutpoint ui`` exports of a 360 project first stitch the recordings they need with
the SDK, when it runs (the command-line ``clips`` never stitches).

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
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint import cache, paths, state
from cutpoint.config import path_key, remap_source, source_path
from cutpoint.insta360 import parse_name
from cutpoint.pano.x3 import X3Clip, assign_ids

logger = logging.getLogger(__name__)

PANO_FILE = "pano.json"
DEFAULT_STYLE = "action-cam"
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
    """``--style`` of ``cutpoint 360 analyze`` when none is given (``pano.default_style``,
    else ``action-cam``). Its activity-specific parts (crash and view captions) come from
    the project's activity profile, so it fits any activity a 360 camera films."""
    return str(pano_config().get("default_style") or DEFAULT_STYLE)


def auto_stitch() -> bool:
    """``cutpoint ui`` stitches a 360 export's recordings first (``pano.auto_stitch``, default on).

    Raises:
        ValueError: If ``pano.auto_stitch`` is set but is not true or false.
    """
    raw = pano_config().get("auto_stitch", True)
    if not isinstance(raw, bool):
        raise ValueError(f"{paths.config_file()}: 'pano.auto_stitch' must be true or false")
    return raw


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
        if a in ("-output", "-inputs"):  # cutpoint sets them; a second one goes unchecked
            raise ValueError(f"{a} is set by cutpoint; remove it from pano.stitch_args")
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


MAX_FOLLOW_DEG = 30.0  # default ``pano.max_follow_deg``


def _number(key: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """``pano.<key>`` (``default`` when unset), checked to be a number within ``[lo, hi]``.

    Raises:
        ValueError: If it is not a number (``null`` and true / false included) or is out
            of range, naming the key and the range.
    """
    cfg = pano_config()
    if key not in cfg:
        return float(default)
    raw = cfg[key]
    ok = isinstance(raw, int | float) and not isinstance(raw, bool) and math.isfinite(raw)
    if ok and (lo is None or raw >= lo) and (hi is None or raw <= hi):
        return float(raw)
    want = (
        f"a number from {lo:g} to {hi:g}"
        if lo is not None and hi is not None
        else f"a number of at least {lo:g}"
        if lo is not None
        else "a number"
    )
    raise ValueError(f"{paths.config_file()}: 'pano.{key}' must be {want}, got {raw!r}")


def level_settings() -> tuple[float, float, float]:
    """``(level_strength, tilt_follow_s, max_follow_deg)`` of the renderer's levelling
    (``pano`` keys; defaults 0.7, 0.5 and :data:`MAX_FOLLOW_DEG`), validated.

    Raises:
        ValueError: If one is not a number, ``level_strength`` is outside 0-1,
            ``tilt_follow_s`` is negative or ``max_follow_deg`` is outside 0-180.
    """
    return (
        _number("level_strength", 0.7, 0.0, 1.0),
        _number("tilt_follow_s", 0.5, 0.0),
        _number("max_follow_deg", MAX_FOLLOW_DEG, 0.0, 180.0),
    )


def cmd_every() -> int:
    """Frames between ``v360`` updates (``pano.cmd_every``, default 1), validated.

    Raises:
        ValueError: If it is not a whole number of at least 1.
    """
    raw = pano_config().get("cmd_every", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError(
            f"{paths.config_file()}: 'pano.cmd_every' must be a whole number of at least 1, "
            f"got {raw!r}"
        )
    return raw


def imu_offset_s() -> float:
    """The renderer's IMU sync (``pano.imu_offset_s``, seconds, default 0), validated.

    Raises:
        ValueError: If it is not a number.
    """
    return _number("imu_offset_s", 0.0)


def equirect_yaw_offset() -> float:
    """Yaw (deg) added for an equirect input (``pano.equirect_yaw_offset``, default 0),
    validated.

    Raises:
        ValueError: If it is not a number.
    """
    return _number("equirect_yaw_offset", 0.0)


# --------------------------------------------------------------------------- pano.json
def save(project: Path, clips: list[X3Clip]) -> Path:
    """Write ``<project>/pano.json`` (merging clips from earlier scans of other folders)."""
    project.mkdir(parents=True, exist_ok=True)
    path = project / PANO_FILE
    old = cache.load_json(path) or {}
    entries: dict[str, tuple[dict[str, Any] | None, X3Clip | None]] = {
        c["lrv"] or c["insv"][0]: (c, None) for c in old.get("clips", [])
    }
    for c in clips:
        entries[c.source] = (None, c)
    # ids over the whole project: a number two recordings have gets the long ids
    objs = [c or X3Clip.from_json(d or {}) for d, c in entries.values()]
    assign_ids(objs)
    by_src = {
        src: c.to_json() if c else {**(d or {}), "clip_id": o.clip_id}
        for (src, (d, c)), o in zip(entries.items(), objs, strict=True)
    }
    for entry in by_src.values():  # an older scan stored the camera's serial: not carried on
        if isinstance(entry.get("meta"), dict):
            entry["meta"] = {k: v for k, v in entry["meta"].items() if k != "serial"}
    cache.save_json(
        path,
        {
            "version": 1,
            "clips": sorted(by_src.values(), key=lambda c: (c["start_local"], c["clip_id"])),
        },
    )
    return path


def load(project: Path) -> dict[str, X3Clip]:
    """``{normalised source path: clip}`` from ``pano.json`` (``{}`` when absent)."""
    data = cache.load_json(project / PANO_FILE)
    if not data:
        return {}
    clips = [X3Clip.from_json(d) for d in data.get("clips", [])]
    assign_ids(clips)  # an older pano.json may give two recordings one number
    for c in clips:  # an older scan paired the files of such recordings by number alone
        lrv, lens = parse_name(c.lrv) if c.lrv else None, parse_name(c.insv[0]) if c.insv else None
        if lrv and lens and (lrv["date"], lrv["time"]) != (lens["date"], lens["time"]):
            _warn_once(
                project / PANO_FILE,
                f"clip {c.clip_id}: the proxy {Path(c.lrv or '').name} and the lens file "
                f"{Path(c.insv[0]).name} are from two recordings; run cutpoint 360 scan "
                f"{Path(c.insv[0]).parent} --out {project} again",
            )
    return {norm(c.source): c for c in clips}


def norm(path: str | Path) -> str:
    """Forward-slash path for lookups (case-folded where its folder is case-insensitive, see
    :func:`path_key`)."""
    return path_key(str(path).replace("\\", "/"))


def clip_for(clips: dict[str, X3Clip], source: str | Path) -> X3Clip | None:
    """The X3 clip whose source identity is ``source`` (also by file name)."""
    hit = clips.get(norm(source))
    if hit is not None:
        return hit
    name, key = source_path(source).name, source_path(norm(source)).name
    for c in clips.values():
        if source_path(c.source).name == name or source_path(norm(c.source)).name == key:
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
_STAMP_RE = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")


def _number_shared(clip: X3Clip) -> bool:
    """Whether another recording in ``clip``'s camera folder has its clip number."""
    f = parse_name(clip.insv[0])
    if f is None:
        return False
    folders = dict.fromkeys([Path(clip.insv[0]).parent, remap_source(clip.insv[0]).parent])
    for d in folders:
        try:
            names = [parse_name(p) for p in d.iterdir()]
        except OSError:
            continue
        if any(
            g and g["num"] == f["num"] and (g["date"], g["time"]) != (f["date"], f["time"])
            for g in names
        ):
            return True
    return False


def find_studio_export(
    clip: X3Clip, extra_dirs: list[Path] | None = None
) -> tuple[Path | None, str]:
    """An Insta360 Studio equirect export of ``clip``, or ``(None, reason)``.

    An export named with a ``YYYYMMDD_HHMMSS`` stamp must carry the clip's; one named with
    the clip number alone is used only when no other recording in the project (its long id)
    or the camera folder has it.
    """
    from cutpoint.errors import FFmpegError  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import probe  # noqa: PLC0415

    f = parse_name(clip.insv[0]) or {}
    stem = f"{f.get('date', '')}_{f.get('time', '')}"
    shared: bool | None = None  # _number_shared, read on the first number-only name
    reasons = []
    for d in _studio_dirs(clip, extra_dirs or []):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".mp4", ".mov") or p.name.lower().endswith(".partial"):
                continue
            if parse_name(p) is not None:
                continue
            stamps = _STAMP_RE.findall(p.stem)
            if stamps:
                if stem not in stamps:
                    continue  # another recording's export
            elif clip.number not in _NUM_RE.findall(p.stem):
                continue
            else:
                if shared is None:  # long id: the project has another recording with it
                    shared = clip.clip_id != clip.number or _number_shared(clip)
                if shared:
                    why = (
                        f"{p.name}: named by clip number {clip.number} only, which another "
                        "recording has too; rename it with the recording's "
                        f"VID_{stem} stem"
                    )
                    _warn_once(p, why)
                    reasons.append(why)
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


def source_flowstate(mode: str | None, inputs: list[str] | tuple[str, ...]) -> bool | None:
    """Whether a render source of ``mode`` reading ``inputs`` is a FlowState equirect:
    False for the dual fisheye (``dfisheye``, the ``proxy`` of director previews), an SDK
    stitch's sidecar (:func:`equirect_flowstate`; None without one) and None for a Studio
    export (assumed FlowState, not recorded) or an unknown mode."""
    if mode in ("dfisheye", "proxy"):
        return False
    if mode == "sdk" and inputs:
        return equirect_flowstate(Path(inputs[0]))
    return None


_flowstate_warned: set[str] = state.register_container(__name__, "_flowstate_warned", set())


def _warn_once(path: Path, message: str) -> None:
    """Log ``message`` as a WARNING once per file (with the FlowState warnings' memory)."""
    key = f"{path}|{message}"
    if key not in _flowstate_warned:
        _flowstate_warned.add(key)
        logger.warning("%s", message)


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
