"""Cutpoint configuration: projects folder, source-path remapping, player tags.

Settings come from ``config.json`` in ``CUTPOINT_HOME`` (or the file named by
``CUTPOINT_CONFIG``; see :mod:`cutpoint.paths`), overridden by environment
variables. ``config.example.json`` in the repository
documents every key.

``projects_dir``
    Where project folders live. ``CUTPOINT_PROJECTS_DIR`` overrides it.
    Default ``<CUTPOINT_HOME>/projects``. Project arguments that
    are a bare name (``my-trip``) or start with ``projects/`` (``projects\\``
    on Windows; both spellings work everywhere) are placed here; absolute
    paths and existing relative paths are used as given.

``source_remap``
    ``{"old prefix": "new prefix"}``. A source path under an old prefix is read
    from the new prefix instead, if the file exists there: point old paths at
    a new location when footage moves (another drive, a backup copy). Projects
    keep recording the original path, so their events, candidates and
    manifests still match.

``source_remap_exclude``
    File-name or full-path globs that are never remapped (e.g. a backup copy
    that is not verified yet).

``stage_dir``
    Local folder for staged copies made by ``cutpoint stage``
    (``CUTPOINT_STAGE_DIR`` overrides it; default
    ``<CUTPOINT_HOME>/stage``). A source on slow or network storage (NAS,
    USB, cloud-synced folder) is copied there once;
    afterwards every ffmpeg/ffprobe read of that source is redirected to the
    local copy (see :func:`cutpoint.stage.staged_path`). Projects still record
    the original path, so deleting the stage folder only makes reads slower again.

``player_tags``
    Your in-game name(s), used by the ``cutpoint events`` HUD OCR to tell your
    kills and deaths apart (fuzzy match, ignoring ``_ . -`` and spaces).
    ``cutpoint events --player TAG`` overrides it per run. Default: none.

``ollama_url``
    Ollama base URL for the frame rater and VLM layers. ``OLLAMA_HOST``
    overrides it. Default ``http://127.0.0.1:11434``
    (see :func:`cutpoint.backends.ollama_url`).

``video_encoder``
    ``auto`` (default), ``nvenc``, ``amf``, ``vaapi`` or ``cpu``. ``CUTPOINT_VIDEO_ENCODER``
    overrides it. Auto picks the first GPU encoder that really encodes, else libx264; a named
    GPU encoder that does not work is an error (see :func:`cutpoint.ffmpeg_utils.video_encoder`).

``vision``
    ``true`` rates the top candidates of ``cutpoint analyze`` and ``cutpoint edit`` with the local
    Ollama vision model unless ``--no-vision`` is given (``--vision`` turns it on for one run).
    Default ``false``: vision is opt-in. ``cutpoint ui`` changes it in Settings.

``compress_peaks``
    ``off`` (default), ``light`` or ``strong``: peak compression and clip leveling of
    ``cutpoint clips``, ``assemble`` and ``edit`` when ``--compress-peaks`` is not given
    (see :data:`cutpoint.encode.PEAK_PRESETS`).

``analysis``
    ``{"skip": [...]}``: analysis steps ``cutpoint analyze`` and ``cutpoint edit`` leave out
    by default, any of ``reactions``, ``keyframes`` and ``crashes`` (``--skip`` overrides it
    per run). Default: none.

``pano``
    Insta360 settings, read by :func:`cutpoint.pano.project.pano_config`.

The analysis cache key is canonicalised through the same table (a remapped
path is keyed as its original), so a remapped source with the same size and
mtime keeps its cache.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

from cutpoint import cache, paths, state
from cutpoint.utils import file_stamp, matches_glob

logger = logging.getLogger(__name__)

VIDEO_ENCODERS = ("auto", "nvenc", "amf", "vaapi", "cpu")
COMPRESS_PEAKS = ("off", "light", "strong")


@dataclass(frozen=True)
class Config:
    """Resolved Cutpoint settings."""

    projects_dir: Path
    source_remap: tuple[tuple[str, str], ...] = ()
    source_remap_exclude: tuple[str, ...] = field(default_factory=tuple)
    stage_dir: Path = field(default_factory=paths.default_stage_dir)
    player_tags: tuple[str, ...] = ()
    ollama_url: str = ""
    video_encoder: str = "auto"
    vision: bool = False
    analysis_skip: tuple[str, ...] = ()
    compress_peaks: str = "off"


def load_config_data(path: Path | None = None) -> dict:
    """The raw ``config.json`` object (``{}`` when the file does not exist).

    Args:
        path: Config file to read (default: :func:`cutpoint.paths.config_file`).

    Raises:
        ValueError: If the file exists but is not valid JSON.
    """
    path = path or paths.config_file()
    if not path.is_file():
        return {}
    try:
        data = cache.read_json(path)
    except FileNotFoundError:  # removed since the check
        return {}
    except ValueError as err:  # its message names the path and the reason
        raise ValueError(f"Invalid config file {err}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file {path}: expected a JSON object")
    return data


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_str_or_list(value: object) -> bool:
    return isinstance(value, str) or _is_str_list(value)


def _is_str_map(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(v, str) for v in value.values())


def _check_type(path: Path, data: dict, key: str, ok: Callable[[object], bool], what: str) -> None:
    """Raise ``ValueError`` naming ``path``, ``key`` and the value's type unless ``data[key]`` is
    unset (missing or null) or passes ``ok``."""
    value = data.get(key)
    if value is None or ok(value):
        return
    got = type(value).__name__
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    bad = [type(v).__name__ for v in items if not isinstance(v, str)]
    if bad:
        got += f" containing {bad[0]}"
    raise ValueError(f"{path}: {key} must be {what}, not {got}")


# The last loaded config, keyed by everything it is read from (see get_config).
_LOADED: dict[tuple, Config] = {}


def _config_key(path: Path) -> tuple:
    """What :func:`get_config` depends on: the file's identity (one stat; None when missing),
    ``CUTPOINT_HOME`` (the default folders) and the ``CUTPOINT_*`` overrides it reads."""
    env = tuple(paths.env(name) for name in ("PROJECTS_DIR", "STAGE_DIR", "VIDEO_ENCODER"))
    return (path, file_stamp(path), paths.home(), *env)


def get_config() -> Config:
    """Load ``config.json`` plus environment overrides.

    The result is cached until the file changes (modification time, size or inode; also when it
    appears or disappears) or a ``CUTPOINT_`` variable it reads changes, so a long-lived process
    (``cutpoint ui``) sees hand edits. ``get_config.cache_clear()`` drops the cache.

    Raises:
        ValueError: Naming the file (or ``CUTPOINT_`` variable) and key, for an invalid file or a
            value of the wrong type or out of range. A failed call is not cached.
    """
    path = paths.config_file()
    key = _config_key(path)
    cfg = _LOADED.get(key)
    if cfg is None:
        cfg = _load_config(path)
        _LOADED.clear()
        _LOADED[key] = cfg
    return cfg


state.register_dict_cache(get_config, _LOADED)


def _load_config(path: Path) -> Config:
    """:func:`get_config` without the cache."""
    data = load_config_data(path)
    for key in ("projects_dir", "stage_dir", "ollama_url", "video_encoder", "compress_peaks"):
        _check_type(path, data, key, _is_str, "a string")
    _check_type(path, data, "source_remap", _is_str_map, "an object mapping strings to strings")
    _check_type(path, data, "source_remap_exclude", _is_str_list, "a list of strings")
    _check_type(path, data, "player_tags", _is_str_or_list, "a list of names")
    _check_type(path, data, "vision", _is_bool, "true or false")
    skip = _analysis_skip(path, data)
    projects = resolve_projects_dir(data)
    remap = data.get("source_remap") or {}
    # Longest prefix first, so nested prefixes win.
    pairs = tuple(sorted(remap.items(), key=lambda kv: -len(kv[0])))
    exclude = tuple(data.get("source_remap_exclude") or ())
    stage = resolve_stage_dir(data)
    tags = data.get("player_tags") or ()
    if isinstance(tags, str):
        tags = (tags,)
    encoder = str(paths.env("VIDEO_ENCODER") or data.get("video_encoder") or "auto")
    encoder = encoder.strip().lower()
    if encoder not in VIDEO_ENCODERS:
        where = "CUTPOINT_VIDEO_ENCODER" if paths.env("VIDEO_ENCODER") else f"{path}: video_encoder"
        raise ValueError(f"{where} is {encoder!r}; expected one of {', '.join(VIDEO_ENCODERS)}")
    peaks = data.get("compress_peaks") or "off"
    if peaks not in COMPRESS_PEAKS:
        raise ValueError(
            f"{path}: compress_peaks is {peaks!r}; expected one of {', '.join(COMPRESS_PEAKS)}"
        )
    return Config(
        projects,
        pairs,
        exclude,
        stage,
        tuple(t for t in tags if t.strip()),
        str(data.get("ollama_url") or ""),
        encoder,
        data.get("vision") is True,
        skip,
        peaks,
    )


def _analysis_skip(path: Path, data: dict) -> tuple[str, ...]:
    """``analysis.skip`` of config ``data``, checked against the steps that can be skipped."""
    from cutpoint.progress import SKIPPABLE  # noqa: PLC0415

    _check_type(path, data, "analysis", lambda v: isinstance(v, dict), "an object")
    section = data.get("analysis") or {}
    _check_type(
        path,
        {"analysis.skip": section.get("skip")},
        "analysis.skip",
        _is_str_list,
        "a list of step names",
    )
    skip = tuple(dict.fromkeys(section.get("skip") or ()))
    bad = [s for s in skip if s not in SKIPPABLE]
    if bad:
        raise ValueError(
            f"{path}: analysis.skip has {bad[0]!r}; expected any of {', '.join(SKIPPABLE)}"
        )
    return skip


def set_config_values(values: dict[str, object], path: Path | None = None) -> Path:
    """Write ``values`` into ``config.json`` atomically, keeping the other keys, their order and
    the file's indentation; the cached :func:`get_config` is dropped. Returns the file written.

    Raises:
        ValueError: If the file exists but is not valid JSON.
        OSError: If it cannot be written.
    """
    path = path or paths.config_file()
    data = load_config_data(path)
    indent: int | str = 2
    with contextlib.suppress(OSError):
        lines = path.read_text(encoding="utf-8").splitlines()
        lead = next((ln[: len(ln) - len(ln.lstrip())] for ln in lines[1:] if ln.strip()), "")
        if lead:
            indent = len(lead) if lead.strip(" ") == "" else lead
    data.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, data, indent=indent)  # type: ignore[arg-type]
    get_config.cache_clear()
    return path


def resolve_projects_dir(data: dict) -> Path:
    """Effective projects folder for config ``data``: env, then ``projects_dir``, then default."""
    return Path(
        paths.env_path("PROJECTS_DIR") or data.get("projects_dir") or paths.default_projects_dir()
    )


def resolve_stage_dir(data: dict) -> Path:
    """Effective stage folder for config ``data``: env, then ``stage_dir``, then default."""
    configured = data.get("stage_dir")
    return paths.env_path("STAGE_DIR") or (
        paths.absolute(configured) if configured else paths.default_stage_dir()
    )


def vision_default() -> bool:
    """Whether ``analyze`` and ``edit`` rate frames with Ollama when neither ``--vision`` nor
    ``--no-vision`` is given (``vision`` in ``config.json``; default False)."""
    return get_config().vision


def analysis_skip() -> tuple[str, ...]:
    """Analysis steps left out when ``--skip`` is not given (``analysis.skip`` in
    ``config.json``; default none)."""
    return get_config().analysis_skip


def compress_peaks_default() -> str:
    """Peak compression preset when ``--compress-peaks`` is not given (``compress_peaks`` in
    ``config.json``; default ``off``)."""
    return get_config().compress_peaks


def player_tags() -> tuple[str, ...]:
    """Configured in-game names (``player_tags`` in ``config.json``; may be empty)."""
    return get_config().player_tags


def projects_dir() -> Path:
    """The configured projects folder (not created here)."""
    return get_config().projects_dir


def resolve_project_path(arg: str | Path, *, bare_names: bool = True) -> Path:
    """Map a CLI path argument onto the projects folder where appropriate.

    Args:
        arg: Path as typed by the user.
        bare_names: Treat a single-component name (``my-trip``) as a project
            under the projects folder. Use False for arguments where a bare
            name means a file in the current folder (e.g. ``--out final.mp4``).

    Returns:
        ``arg`` unchanged if it is absolute or exists relative to the current
        folder; ``projects_dir/<rest>`` if it starts with ``projects``
        (``projects/<name>`` or ``projects\\<name>``, on any platform);
        ``projects_dir/<name>`` for a bare name (when ``bare_names``);
        otherwise ``arg`` unchanged.
    """
    p = Path(arg)
    if p.is_absolute() or p.drive or p.exists():
        return p
    parts = p.parts
    if sys.platform != "win32" and "\\" in str(arg):  # projects\<name> typed on POSIX
        parts = PurePosixPath(str(arg).replace("\\", "/")).parts
    if parts and parts[0].lower() == "projects":
        return projects_dir().joinpath(*parts[1:])
    if bare_names and len(parts) == 1 and parts[0] not in (".", ".."):
        return projects_dir() / parts[0]
    return p


# -- path keys ---------------------------------------------------------------
_WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:")


def path_key(path: str) -> str:
    """``path`` as a lookup key: lower-cased on Windows (case-insensitive file
    systems), unchanged elsewhere."""
    return path.lower() if sys.platform == "win32" else path


def is_absolute_any(path: str | PurePath) -> bool:
    """``path`` is absolute here or as a Windows path (``D:\\x``, ``D:/x``, ``\\\\server\\share``).

    A project written on Windows records ``D:\\Videos\\x.mp4``; on POSIX that string
    is not :meth:`Path.is_absolute`, but it must not be joined to a folder either.
    """
    return Path(path).is_absolute() or PureWindowsPath(str(path)).is_absolute()


def source_path(path: str | PurePath) -> PurePath:
    """A stored source path as a pure path, for its ``name`` / ``stem``.

    Projects written on Windows record ``D:\\Videos\\x.mp4``; on POSIX that
    string is parsed as a Windows path (drive letter, or backslashes and no
    ``/``), so ``.name`` is ``x.mp4`` there too. Other paths use the platform's
    rules.
    """
    text = str(path)
    if (
        sys.platform == "win32"
        or _WINDOWS_PATH_RE.match(text)
        or ("\\" in text and "/" not in text)
    ):
        return PureWindowsPath(text)
    return PurePosixPath(text)


# -- source remapping ------------------------------------------------------
def _pure(path: str) -> PurePath:
    """``path`` with the platform's separators (backslashes on Windows)."""
    return PureWindowsPath(path) if sys.platform == "win32" else PurePosixPath(path)


def _sep() -> str:
    return "\\" if sys.platform == "win32" else "/"


def norm_path(path: str) -> str:
    """Form of ``path`` for prefix comparison.

    Windows: case-folded and backslash-separated. Elsewhere: case kept, POSIX
    separators.
    """
    if sys.platform == "win32":
        return str(PureWindowsPath(path)).replace("/", "\\").rstrip("\\").lower()
    return str(PurePosixPath(path)).rstrip("/")


def is_foreign_windows(path: str) -> bool:
    """POSIX: ``path`` is absolute only as a Windows path (``D:\\x``, ``\\\\server\\share``),
    as recorded by a project written on Windows. False on Windows."""
    return sys.platform != "win32" and is_absolute_any(path) and not Path(path).is_absolute()


def windows_norm(path: str) -> str:
    """:func:`norm_path` with Windows rules on any platform: case-folded, backslashes."""
    return str(PureWindowsPath(path)).replace("/", "\\").rstrip("\\").lower()


_DRIVE_RE = re.compile(r"^([A-Za-z]:)(?:[\\/]|$)")
_ERROR_MORE_DATA = 234
_ERROR_CONNECTION_UNAVAIL = 1201  # remembered mapping, not connected; the name is still returned
_ERROR_NOT_CONNECTED = 2250  # the letter is not a network drive


@state.register_cache
@functools.cache
def _drive_remote(drive: str) -> str | None:
    """Windows: the UNC root (``\\\\server\\share``) mapped drive ``drive`` (``Z:``)
    points to, or None.

    Asks ``WNetGetConnectionW``, which also names a remembered mapping that is
    disconnected, then the persistent mapping in ``HKCU\\Network\\<letter>``
    (skipped when the API says the letter is not a network drive).
    Never raises; None on other platforms.
    """
    if sys.platform != "win32":
        return None
    rc = None
    try:
        import ctypes  # noqa: PLC0415 - Windows only
    except ImportError:
        pass
    else:
        try:
            get_connection = ctypes.WinDLL("mpr").WNetGetConnectionW  # type: ignore[attr-defined]
            size = ctypes.c_ulong(1024)
            for _ in range(2):  # retry once with the size ERROR_MORE_DATA asked for
                buf = ctypes.create_unicode_buffer(size.value)
                rc = get_connection(drive, buf, ctypes.byref(size))
                if rc in (0, _ERROR_CONNECTION_UNAVAIL) and buf.value.startswith("\\\\"):
                    return buf.value
                if rc != _ERROR_MORE_DATA:
                    break
        except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
            pass
    if rc == _ERROR_NOT_CONNECTED:  # not a network drive; ignore stale registry entries
        return None
    try:
        import winreg  # noqa: PLC0415 - Windows only
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, f"Network\\{drive[0]}") as key:
            remote = str(winreg.QueryValueEx(key, "RemotePath")[0])
    except (OSError, ValueError):
        return None
    return remote if remote.startswith("\\\\") else None


def unc_path(path: str) -> str:
    """Windows: ``path`` on a mapped network drive in UNC form, also while the
    drive is disconnected (``Z:\\F\\a.mp4`` -> ``\\\\server\\share\\F\\a.mp4``).

    That is the form ``Path.resolve()`` gives a reachable file there. Unchanged
    on other platforms, for local drives and UNC paths, and when the mapping
    cannot be read.
    """
    if sys.platform != "win32" or not (m := _DRIVE_RE.match(path)):
        return path
    remote = _drive_remote(m.group(1).upper())
    if not remote:
        return path
    return str(PureWindowsPath(remote, path[2:].lstrip("\\/")))


@state.register_cache
@functools.cache
def _prefix_forms(prefix: str) -> tuple[str, ...]:
    """A prefix as typed plus its resolved and UNC forms (mapped drive -> UNC).

    A Windows prefix on POSIX (``D:\\Videos``) is compared the Windows way
    (:func:`windows_norm`); making it absolute here would prepend the current folder.
    """
    if is_foreign_windows(prefix):
        return (windows_norm(prefix),)
    absolute = os.path.abspath(prefix)  # noqa: PTH100 - normalises '..', keeps drive letters
    forms = {norm_path(absolute), norm_path(unc_path(absolute))}
    with contextlib.suppress(OSError):
        forms.add(norm_path(str(Path(prefix).resolve())))
    return tuple(forms)


def _strip_prefix(path: str, prefix: str) -> str | None:
    """The remainder of ``path`` (as typed or in UNC form) below ``prefix`` (any
    form), or None."""
    for text in dict.fromkeys((path, unc_path(path))):
        if is_foreign_windows(text):  # D:\Videos\x.mp4 on POSIX: rest with POSIX separators
            low, sep, pure = windows_norm(text), "\\", str(PureWindowsPath(text))
        else:
            low, sep, pure = norm_path(text), _sep(), str(_pure(text))
        for form in _prefix_forms(prefix):
            if low == form or low.startswith(form + sep):
                rest = pure[len(form) :].lstrip("\\/")
                return rest.replace("\\", "/") if sep != _sep() else rest
    return None


def _remap_excluded(path: str) -> bool:
    """``path`` matches ``source_remap_exclude``, comparing the mapped-drive and
    UNC spellings (:func:`unc_path`) of both the path and the globs, so ``Z:\\...``
    and ``\\\\server\\share\\...`` exclude each other."""
    patterns = get_config().source_remap_exclude
    if not patterns:
        return False
    globs = list(dict.fromkeys(g for pat in patterns for g in (pat, unc_path(pat))))
    return any(matches_glob(norm_path(text), globs) for text in {path, unc_path(path)})


def remap_source(path: str | Path) -> Path:
    """Where to read a source from: its ``source_remap`` target if that exists.

    Falls back to the original path when no prefix matches, the file is in
    ``source_remap_exclude``, or the remapped file does not exist or cannot be
    checked (permission denied, stale mount, I/O error, name too long).
    """
    original = str(path)
    if _remap_excluded(original):
        return Path(path)
    for old, new in get_config().source_remap:
        rest = _strip_prefix(original, old)
        if rest is None:
            continue
        target = Path(new) / rest if rest else Path(new)
        try:
            found = target.exists()
        except OSError as err:  # exists() only swallows ENOENT/ENOTDIR/EBADF/ELOOP
            logger.debug("remap target not reachable (%s), using original: %s", err, target)
            break
        if found:
            logger.debug("remap %s -> %s", original, target)
            return target
        logger.debug("remap target missing, using original: %s", target)
        break
    return Path(path)


def canonical_source(path: str | Path) -> str:
    """The original (pre-remap) spelling of ``path``, for cache keys.

    A path under a remap *target* is expressed under its *source* prefix, so
    ``E:\\Footage-copy\\x.mp4`` keys like ``D:\\Footage\\x.mp4`` when the table maps
    ``D:\\Footage`` to ``E:\\Footage-copy``. A source prefix on a mapped drive is
    given in UNC form (:func:`unc_path`), as a resolved original is.
    """
    text = str(path)
    for old, new in get_config().source_remap:
        rest = _strip_prefix(text, new)
        if rest is not None:
            base = PureWindowsPath(old) if is_foreign_windows(old) else _pure(old)
            return unc_path(str(base / rest) if rest else str(base))
    return text
