"""Optional dependency groups (``pyproject.toml`` extras) and how to install them.

Used to turn a missing optional import into an actionable message such as
"telemetry_parser is not installed; it comes with the 'ski' extra: ...".
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Top-level import name -> extra that provides it.
MODULE_EXTRA: dict[str, str] = {
    "telemetry_parser": "ski",
    "torch": "events",
    "torchvision": "events",
    "open_clip": "events",
    "panns_inference": "events",
    "torchlibrosa": "events",
    "rapidocr_onnxruntime": "events",
    "onnxruntime": "events",
    "scipy": "events",
    "sklearn": "events",
}


def uv_managed() -> bool:
    """True when this environment should be changed with uv rather than pip.

    That is the case under ``uv run`` (uv sets ``UV``) or when the environment
    has no pip at all (``uv venv`` / ``uv sync`` do not install it).
    """
    return bool(os.environ.get("UV")) or importlib.util.find_spec("pip") is None


def install_hint(extra: str) -> str:
    """The command that adds ``extra`` to the running environment."""
    if uv_managed():
        return (
            f"uv sync --extra {extra} (list every other extra you use too: uv sync "
            f"removes packages of extras you leave out), or: uv pip install --python "
            f'"{sys.executable}" -e ".[{extra}]"'
        )
    return f'"{sys.executable}" -m pip install -e ".[{extra}]" (from the repository)'


def missing_module_message(module: str) -> str | None:
    """Message for a missing optional module, or None if it is not an extra's."""
    top = module.split(".", 1)[0]
    extra = MODULE_EXTRA.get(top)
    if extra is None:
        return None
    return f"{top} is not installed; it comes with the '{extra}' extra: {install_hint(extra)}"
