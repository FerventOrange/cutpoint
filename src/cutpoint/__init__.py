"""Cutpoint: local AI highlight finding, clipping and assembly (analyze -> clips -> assemble).

:func:`reset_state` clears every per-process cache (config, stage index, tool
lookups, ...) for long-lived processes that change settings between jobs.
"""

from importlib.metadata import PackageNotFoundError, version

from cutpoint.state import reset_state

try:
    __version__ = version("cutpoint")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "reset_state"]
