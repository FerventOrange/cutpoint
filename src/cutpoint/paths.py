"""Where Cutpoint keeps its data: one ``CUTPOINT_HOME`` folder outside the package.

``CUTPOINT_HOME`` defaults to ``%LOCALAPPDATA%\\cutpoint`` on Windows and
``$XDG_DATA_HOME/cutpoint`` (``~/.local/share/cutpoint``) elsewhere; the
``CUTPOINT_HOME`` environment variable overrides it. Layout::

    CUTPOINT_HOME/
      config.json     settings (see config.example.json in the repository)
      cache/          per-source analysis cache (audio, transcripts, scenes, ...)
      models/         PANNs, CLIP, detector, aesthetic head, Silero VAD, LLMs
      tools/          llama.cpp, CLIProxyAPI binary (``tools/cliproxy/``)
      cliproxy/       CLIProxyAPI keys, config, and subscription auth records
      stage/          local copies of sources on slow/network storage (``cutpoint stage``)
      styles/         learned style presets (``cutpoint refmatch`` / ``fit``)
      presets/        HUD presets written by region discovery (``cutpoint events``)
      projects/       project folders, unless ``projects_dir`` says otherwise

Every location can be overridden on its own by an environment variable:

==================  ======================
location            variable
==================  ======================
config file         CUTPOINT_CONFIG
cache               CUTPOINT_CACHE_DIR
models              CUTPOINT_MODELS_DIR
tools               CUTPOINT_TOOLS_DIR
stage               CUTPOINT_STAGE_DIR
projects            CUTPOINT_PROJECTS_DIR
learned styles      CUTPOINT_STYLES_DIR
==================  ======================

``stage_dir`` and ``projects_dir`` in ``config.json`` sit between the
environment variables and the built-in defaults (see :mod:`cutpoint.config`).

This module has no package-internal imports, so anything may import it. Nothing
here creates folders.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "cutpoint"
PACKAGE_DIR = Path(__file__).resolve().parent
PRESETS_DIR = PACKAGE_DIR / "events" / "presets"  # bundled HUD presets (read-only data)


def absolute(value: str | os.PathLike[str]) -> Path:
    """``value`` with ``~`` expanded, made absolute against the current folder.

    Uses :func:`os.path.abspath` (no symlink or mapped-drive resolution, so a
    path on a mapped network drive keeps its drive letter instead of turning
    into its UNC form).
    """
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))  # noqa: PTH100, PTH111 - see docstring


def env(name: str) -> str | None:
    """Value of ``CUTPOINT_<name>``, else None.

    Empty values count as unset.

    Args:
        name: Variable name without prefix, e.g. ``"CACHE_DIR"``.
    """
    return os.environ.get("CUTPOINT_" + name) or None


def env_path(name: str) -> Path | None:
    """:func:`env` as an absolute path (None when unset)."""
    value = env(name)
    return absolute(value) if value else None


def default_home() -> Path:
    """The platform default for ``CUTPOINT_HOME`` (ignores the variable)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return absolute(Path(base) / APP_NAME)


def home() -> Path:
    """``CUTPOINT_HOME``: the environment variable or :func:`default_home` (absolute)."""
    value = os.environ.get("CUTPOINT_HOME")
    return absolute(value) if value else default_home()


def _dir(name: str, default: str) -> Path:
    return env_path(name) or home() / default


def config_file() -> Path:
    """``config.json``: ``CUTPOINT_CONFIG`` or ``<home>/config.json``."""
    return _dir("CONFIG", "config.json")


def cache_dir() -> Path:
    """Analysis cache: ``CUTPOINT_CACHE_DIR`` or ``<home>/cache``."""
    return _dir("CACHE_DIR", "cache")


def models_dir() -> Path:
    """Model weights: ``CUTPOINT_MODELS_DIR`` or ``<home>/models``."""
    return _dir("MODELS_DIR", "models")


def tools_dir() -> Path:
    """External tools (llama.cpp): ``CUTPOINT_TOOLS_DIR`` or ``<home>/tools``."""
    return _dir("TOOLS_DIR", "tools")


def styles_dir() -> Path:
    """Learned style presets: ``CUTPOINT_STYLES_DIR`` or ``<home>/styles``."""
    return _dir("STYLES_DIR", "styles")


def presets_dir() -> Path:
    """User HUD presets (written by region discovery): ``<home>/presets``.

    Lookup order for a game's preset: ``<project>/hud_presets``, then this
    folder, then the bundled :data:`PRESETS_DIR`.
    """
    return home() / "presets"


def profiles_dir() -> Path:
    """Activity profile library (learned / synthesized profiles): ``<home>/profiles``."""
    return home() / "profiles"


def default_stage_dir() -> Path:
    """Stage folder when neither the environment nor ``config.json`` sets one."""
    return home() / "stage"


def default_projects_dir() -> Path:
    """Projects folder when neither the environment nor ``config.json`` sets one."""
    return home() / "projects"
