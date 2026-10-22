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

``pano``
    Insta360 settings, read by :func:`cutpoint.pano.project.pano_config`.

The analysis cache key is canonicalised through the same table (a remapped
path is keyed as its original), so a remapped source with the same size and
mtime keeps its cache.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

from cutpoint import paths, state
from cutpoint.utils import matches_glob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Resolved Cutpoint settings."""

    projects_dir: Path
    source_remap: tuple[tuple[str, str], ...] = ()
    source_remap_exclude: tuple[str, ...] = field(default_factory=tuple)
    stage_dir: Path = field(default_factory=paths.default_stage_dir)
    player_tags: tuple[str, ...] = ()
    ollama_url: str = ""


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
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Invalid config file {path}: {err}") from None
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file {path}: expected a JSON object")
    return data


@state.register_cache
@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load ``config.json`` plus environment overrides (cached per process)."""
    path = paths.config_file()
    data = load_config_data(path)
    projects = (
        paths.env_path("PROJECTS_DIR") or data.get("projects_dir") or paths.default_projects_dir()
    )
    remap = data.get("source_remap") or {}
    if not isinstance(remap, dict):
        raise ValueError(f"{path}: source_remap must be an object of prefix pairs")
    # Longest prefix first, so nested prefixes win.
    pairs = tuple(sorted(((str(k), str(v)) for k, v in remap.items()), key=lambda kv: -len(kv[0])))
    exclude = tuple(str(p) for p in data.get("source_remap_exclude") or ())
    stage = resolve_stage_dir(data)
    tags = data.get("player_tags") or ()
    if isinstance(tags, str):
        tags = (tags,)
    if not isinstance(tags, (list, tuple)):
        raise ValueError(f"{path}: player_tags must be a list of names")
    return Config(
        Path(projects),
        pairs,
        exclude,
        stage,
        tuple(str(t) for t in tags if str(t).strip()),
        str(data.get("ollama_url") or ""),
    )


def resolve_stage_dir(data: dict) -> Path:
    """Effective stage folder for config ``data``: env, then ``stage_dir``, then default."""
    configured = data.get("stage_dir")
    return paths.env_path("STAGE_DIR") or (
        paths.absolute(configured) if configured else paths.default_stage_dir()
    )


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


_DRIVE_RE = re.compile(r"^([A-Za-z]:)(?:[\\/]|$)")
_ERROR_MORE_DATA = 234
_ERROR_CONNECTION_UNAVAIL = 1201  # remembered mapping, not connected; the name is still returned


@state.register_cache
@functools.cache
def _drive_remote(drive: str) -> str | None:
    """Windows: the UNC root (``\\\\server\\share``) mapped drive ``drive`` (``Z:``)
    points to, or None.

    Asks ``WNetGetConnectionW``, which also names a remembered mapping that is
    disconnected, then the persistent mapping in ``HKCU\\Network\\<letter>``.
    Never raises; None on other platforms.
    """
    if sys.platform != "win32":
        return None
    import ctypes  # noqa: PLC0415 - Windows only
    import winreg  # noqa: PLC0415 - Windows only

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
    """A prefix as typed plus its resolved and UNC forms (mapped drive -> UNC)."""
    absolute = os.path.abspath(prefix)  # noqa: PTH100 - normalises '..', keeps drive letters
    forms = {norm_path(absolute), norm_path(unc_path(absolute))}
    with contextlib.suppress(OSError):
        forms.add(norm_path(str(Path(prefix).resolve())))
    return tuple(forms)


def _strip_prefix(path: str, prefix: str) -> str | None:
    """The remainder of ``path`` (as typed or in UNC form) below ``prefix`` (any
    form), or None."""
    sep = _sep()
    for text in dict.fromkeys((path, unc_path(path))):
        low = norm_path(text)
        for form in _prefix_forms(prefix):
            if low == form or low.startswith(form + sep):
                return str(_pure(text))[len(form) :].lstrip("\\/")
    return None


def _remap_excluded(path: str) -> bool:
    return matches_glob(norm_path(path), get_config().source_remap_exclude)


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
            return unc_path(str(_pure(old) / rest) if rest else str(_pure(old)))
    return text
