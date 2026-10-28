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

``export_workers``
    Clips ``cutpoint clips`` (and ``edit``) export at once: ``auto`` (default: 2 with a GPU
    encoder, 1 with libx264, which already uses every core) or a number, clamped to 1-3 (NVENC
    session limits). ``CUTPOINT_EXPORT_WORKERS`` overrides it. A run with a 360 clip exports
    one clip at a time.

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

``assistant``
    The ``cutpoint ui`` assistant: ``provider`` (``ollama``, the default; ``openai`` for OpenAI
    or a server that speaks its API; ``anthropic``), ``model`` (default: Ollama ``qwen3.5:9b``
    when it is pulled, else the vision model ``CUTPOINT_VISION_MODEL`` or ``qwen3-vl:8b``;
    Anthropic ``claude-sonnet-5-5``; OpenAI-compatible: the server's only model), ``url`` (the
    base URL; default: the Ollama URL above, ``https://api.openai.com/v1`` or
    ``https://api.anthropic.com``), ``context_tokens`` (the model's context size; default
    8192 on this machine, 32768 hosted) and ``consent`` (``<provider>@<host>`` of the hosted
    providers the user agreed to send project text to). API keys are never read from or
    written to this file (see :mod:`cutpoint.assistant.keys`).

The analysis cache key is canonicalised through the same table (a remapped
path is keyed as its original), so a remapped source with the same size and
mtime keeps its cache.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import logging
import os
import re
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

from cutpoint import cache, paths, state
from cutpoint.utils import file_stamp, matches_glob

logger = logging.getLogger(__name__)

VIDEO_ENCODERS = ("auto", "nvenc", "amf", "vaapi", "cpu")
COMPRESS_PEAKS = ("off", "light", "strong")
MAX_EXPORT_WORKERS = 3  # NVENC sessions: consumer GeForce cards allow few at once
ASSISTANT_PROVIDERS = ("ollama", "openai", "anthropic")
MIN_CONTEXT_TOKENS = 2048  # room for the rules, a few moments and one reply


@dataclass(frozen=True)
class AssistantSettings:
    """The ``assistant`` section of ``config.json`` (see :mod:`cutpoint.assistant`)."""

    provider: str = "ollama"  # one of ASSISTANT_PROVIDERS
    model: str = ""  # "": the provider's default (Ollama: qwen3.5:9b if pulled, else vision)
    url: str = ""  # base URL; "": the provider's (Ollama: cutpoint.backends.ollama_url)
    context_tokens: int = 0  # 0: the provider's default (8192 on this machine, 32768 hosted)
    consent: tuple[str, ...] = ()  # "<provider>@<host>" the user agreed to send project text to


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
    export_workers: int | None = None  # None: auto
    assistant: AssistantSettings = field(default_factory=AssistantSettings)


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
    names = ("PROJECTS_DIR", "STAGE_DIR", "VIDEO_ENCODER", "EXPORT_WORKERS")
    env = tuple(paths.env(name) for name in names)
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
        _export_workers(path, data),
        _assistant(path, data),
    )


def _export_workers(path: Path, data: dict) -> int | None:
    """``export_workers`` (``CUTPOINT_EXPORT_WORKERS`` first) clamped to 1-3; None for auto."""
    env = paths.env("EXPORT_WORKERS")
    value = env if env is not None else data.get("export_workers", "auto")
    where = "CUTPOINT_EXPORT_WORKERS" if env is not None else f"{path}: export_workers"
    if isinstance(value, str) and value.strip().lower() in ("", "auto"):
        return None
    try:
        if isinstance(value, bool | float):  # int() would accept them
            raise ValueError
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} is {value!r}; expected auto or a number of clips") from None
    return min(max(n, 1), MAX_EXPORT_WORKERS)


def _assistant(path: Path, data: dict) -> AssistantSettings:
    """The ``assistant`` section of config ``data``, checked."""
    _check_type(path, data, "assistant", lambda v: isinstance(v, dict), "an object")
    section = data.get("assistant") or {}
    checks = (
        ("provider", _is_str, "a string"),
        ("model", _is_str, "a string"),
        ("url", _is_str, "a string"),
        ("consent", _is_str_list, "a list of <provider>@<host> entries"),
    )
    for key, ok, what in checks:
        _check_type(path, {f"assistant.{key}": section.get(key)}, f"assistant.{key}", ok, what)
    provider = (section.get("provider") or "ollama").strip().lower()
    if provider not in ASSISTANT_PROVIDERS:
        raise ValueError(
            f"{path}: assistant.provider is {provider!r}; "
            f"expected one of {', '.join(ASSISTANT_PROVIDERS)}"
        )
    tokens = section.get("context_tokens")
    if tokens is None:
        tokens = 0  # the provider's default
    elif isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < MIN_CONTEXT_TOKENS:
        raise ValueError(
            f"{path}: assistant.context_tokens is {tokens!r}; "
            f"expected a whole number of at least {MIN_CONTEXT_TOKENS}"
        )
    return AssistantSettings(
        provider,
        (section.get("model") or "").strip(),
        (section.get("url") or "").strip(),
        tokens,
        tuple(dict.fromkeys(section.get("consent") or ())),
    )


def assistant_settings() -> AssistantSettings:
    """The ``assistant`` section of ``config.json`` (defaults when it is missing)."""
    return get_config().assistant


def export_workers() -> int | None:
    """Clips exported at once (``export_workers`` in ``config.json``); None for auto."""
    return get_config().export_workers


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


_FOLDS_CASE_MAX = 256  # folders remembered by folds_case; emptied when full
# (st_dev, st_ino) of a folder (st_dev and its path without inode numbers) -> folds case
_FOLDS_CASE: dict[tuple[int, int | str], bool] = state.register_container(
    __name__, "_FOLDS_CASE", {}
)
# folder as given (made absolute) -> folds case, so a repeat call does no I/O at all
_FOLDS_CASE_DIRS: dict[str, bool] = state.register_container(__name__, "_FOLDS_CASE_DIRS", {})


def _same_entry(a: os.stat_result, b: os.stat_result) -> bool | None:
    """``a`` and ``b`` are one file (None: no inode numbers to tell)."""
    if not a.st_ino or not b.st_ino:
        return None
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _swapped(name: str) -> str | None:
    """``name`` with its ASCII letters in the other case, or None if it has none (non-ASCII
    names are skipped: their case mapping and Unicode normalisation vary per file system)."""
    return name.swapcase() if name.isascii() and name.swapcase() != name else None


def _probe_case(folder: Path) -> bool | None:
    """Whether the existing folder ``folder`` (resolved) is on a case-insensitive file system.

    Read-only: a case-swapped spelling of an entry in it (the first 100 are tried) or of
    the folder's own name (looked up in its parent, unless the folder is a mount point)
    either is the same file (True) or is missing or another file (False). None if no name
    there has ASCII letters, there are no inode numbers to compare or the folder cannot be
    read.
    """
    try:
        with os.scandir(folder) as entries:
            for entry in itertools.islice(entries, 100):  # a huge folder: not all of it
                if (swapped := _swapped(entry.name)) is None:
                    continue
                try:
                    est = entry.stat(follow_symlinks=False)
                    other = (folder / swapped).lstat()
                except FileNotFoundError:
                    if (folder / entry.name).is_symlink() or (folder / entry.name).exists():
                        return False  # the swapped spelling is missing
                    continue  # the entry itself went away
                except OSError:
                    continue
                if (same := _same_entry(other, est)) is not None:
                    return same
    except OSError:
        pass
    if (swapped := _swapped(folder.name)) is None:
        return None
    try:
        st = folder.stat()
        if folder.parent.stat().st_dev != st.st_dev:  # a mount point: the name is on another FS
            return None
    except OSError:
        return None
    try:
        other = (folder.parent / swapped).stat()
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return _same_entry(other, st)


def folds_case(folder: str | PurePath) -> bool:
    """Whether names in ``folder`` are case-insensitive (always True on Windows).

    Elsewhere the nearest existing folder of ``folder`` and its parents is checked
    (:func:`_probe_case`, without creating files), then its parents on the same file
    system while no name there tells. Unknown counts as case-sensitive.

    The answer is remembered per folder string for the life of the process (until
    :func:`~cutpoint.state.reset_state`), so a repeat call does no I/O and a missing or
    offline folder is walked up once, not per call. A folder that does not exist yet (an
    output folder created later) is assumed to follow its nearest existing parent, before
    and after it is created. The probe result is also remembered per existing folder
    (device and inode), so other spellings of it are not probed again.
    """
    if sys.platform == "win32":
        return True
    name = os.fspath(folder)
    if not os.path.isabs(name):  # noqa: PTH117 - no Path: a cache hit stays cheap
        name = os.path.abspath(name)  # noqa: PTH100 - no I/O
    if (hit := _FOLDS_CASE_DIRS.get(name)) is not None:
        return hit
    if len(_FOLDS_CASE_DIRS) >= _FOLDS_CASE_MAX:
        _FOLDS_CASE_DIRS.clear()
    _FOLDS_CASE_DIRS[name] = found = _folds_case_uncached(Path(name))
    return found


def _folds_case_uncached(start: Path) -> bool:
    """:func:`folds_case` without the per-string cache."""
    for d in (start, *start.parents):
        try:
            st = d.stat()
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode):
            continue
        break
    else:
        return False
    key = (st.st_dev, st.st_ino or os.path.abspath(d))  # noqa: PTH100 - no I/O
    if (hit := _FOLDS_CASE.get(key)) is not None:
        return hit
    found = None
    with contextlib.suppress(OSError):
        real = d.resolve()  # the folder a symlink points to
        for p in (real, *real.parents):
            if p.stat().st_dev != st.st_dev:
                break
            if (found := _probe_case(p)) is not None:
                break
    if len(_FOLDS_CASE) >= _FOLDS_CASE_MAX:
        _FOLDS_CASE.clear()
    _FOLDS_CASE[key] = bool(found)
    return bool(found)


def path_key(path: str, folder: str | PurePath | None = None) -> str:
    """``path`` as a lookup key: lower-cased where names are case-insensitive, unchanged
    elsewhere.

    Windows: always lower-cased. Elsewhere (macOS APFS/HFS+, exFAT or FAT drives on Linux,
    NTFS only when mounted case-insensitive) it is lower-cased if :func:`folds_case` says so
    for ``folder`` (the folder a bare name such as ``001_A.mp4`` is in) or, without
    ``folder``, for the folder of ``path``. The whole string is folded even if only its last
    folder is case-insensitive, so compare only keys made for the same folder or for paths
    in the same file. A Windows-form path on POSIX (``D:\\x.mp4``, a project written on
    Windows) is not folded without ``folder``.
    """
    if sys.platform == "win32":
        return path.lower()
    if folder is None:
        if not os.path.isabs(path) and is_foreign_windows(path):  # noqa: PTH117 - hot loop
            return path
        folder = os.path.dirname(path)  # noqa: PTH120 - no Path: this runs in loops
    return path.lower() if folds_case(folder) else path


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
