"""Registry of process-global state, cleared by :func:`cutpoint.reset_state`.

Cutpoint memoises a few things per process: the parsed ``config.json``, the
stage index, tool lookups, per-clip scores, "already warned" sets and loaded
models. One CLI run never notices, but a long-lived process (the Python
API, the test suite) that changes ``CUTPOINT_HOME``, the
environment or ``config.json`` between jobs must forget them first.

Every module that keeps such state registers it here when it is imported::

    @state.register_cache
    @functools.cache
    def find_tool(name: str) -> str: ...

    _SEEN: set[str] = state.register_container(__name__, "_SEEN", set())

Loaded ML models are registered with ``model=True``. They are keyed by
everything that selects them (architecture, weights), so a config change
loads a new one anyway; :func:`reset_state` keeps them unless
``models=True``, because reloading costs seconds and VRAM.
"""

from __future__ import annotations

from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Any, Protocol


class _Cached(Protocol):
    def cache_clear(self) -> None: ...
    def cache_info(self) -> Any: ...


class _Clearable(Sized, Protocol):
    def clear(self) -> None: ...


@dataclass(frozen=True)
class Entry:
    """One registered piece of state."""

    name: str
    clear: Callable[[], None]
    size: Callable[[], int]
    model: bool = False


_REGISTRY: dict[str, Entry] = {}


def register(entry: Entry) -> None:
    """Add (or replace, on module reload) an entry."""
    _REGISTRY[entry.name] = entry


def register_cache[F: _Cached](fn: F) -> F:
    """Register an ``functools.cache`` / ``lru_cache`` function; returns it (decorator)."""
    name = f"{fn.__module__}.{fn.__qualname__}"  # type: ignore[attr-defined]
    register(Entry(name, fn.cache_clear, lambda: fn.cache_info().currsize))
    return fn


def register_container[C: _Clearable](module: str, attr: str, obj: C, *, model: bool = False) -> C:
    """Register a module-level dict/set (``module.attr``) that is emptied on reset."""
    register(Entry(f"{module}.{attr}", obj.clear, lambda: len(obj), model))
    return obj


def reset_state(*, models: bool = False) -> None:
    """Forget every per-process cache of the imported Cutpoint modules.

    Call it after changing ``CUTPOINT_HOME``, ``CUTPOINT_*`` / ``OLLAMA_HOST``
    variables or ``config.json`` in a process that already ran a job.

    Not thread-safe: call it between jobs, never while a job runs in another
    thread (the caches are plain dicts and ``functools`` caches).

    Args:
        models: Also drop loaded ML models (they are keyed by architecture
            and weights, so keeping them is safe; dropping frees memory).
    """
    for entry in list(_REGISTRY.values()):
        if models or not entry.model:
            entry.clear()
