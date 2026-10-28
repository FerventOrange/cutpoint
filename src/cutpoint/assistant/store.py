"""The assistant's files in ``<project>/ui/assistant``.

``chat.json``
    ``{"turns": [...]}``: the finished turns, oldest first (:meth:`cutpoint.assistant.session.Turn.record`).
    Clear chat empties it.
``history.json``
    ``{"applied": [...]}``: each proposal the user applied, with the sequence before and after,
    so it can be reverted after a reload. Clear chat keeps it.

Writes are atomic (:func:`cutpoint.cache.save_json`); a file a link takes outside the project is
never read or written.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from cutpoint import cache

FOLDER = "ui/assistant"
CHAT = "chat.json"
HISTORY = "history.json"
MAX_TURNS = 200  # the oldest go first
MAX_APPLIED = 500

_LOCK = threading.Lock()  # one read-modify-write at a time (turn threads and requests)


class StoreError(RuntimeError):
    """A file of the store resolves outside the project folder."""


def _file(p: Path, name: str) -> Path:
    f = p / FOLDER / name
    try:
        inside = f.resolve().is_relative_to(p.resolve())
    except (OSError, RuntimeError):  # RuntimeError: a symlink loop
        inside = False
    if not inside:
        raise StoreError(f"{FOLDER}/{name} resolves outside the project folder; remove the link")
    return f


def _items(p: Path, name: str, key: str) -> list[dict[str, Any]]:
    try:
        data = cache.load_json(_file(p, name))
    except StoreError:
        return []
    items = data.get(key) if isinstance(data, dict) else None
    return [x for x in items or [] if isinstance(x, dict)]


def turns(p: Path) -> list[dict[str, Any]]:
    """The saved chat turns, oldest first."""
    return _items(p, CHAT, "turns")


def add_turn(p: Path, record: dict[str, Any]) -> None:
    with _LOCK:
        kept = [*turns(p), record][-MAX_TURNS:]
        cache.save_json(_file(p, CHAT), {"turns": kept})


def clear_chat(p: Path) -> None:
    """Forget the chat; the history of applied proposals stays."""
    with _LOCK:
        cache.save_json(_file(p, CHAT), {"turns": []})


def history(p: Path) -> list[dict[str, Any]]:
    """The applied proposals, oldest first."""
    return _items(p, HISTORY, "applied")


def add_applied(p: Path, entry: dict[str, Any]) -> None:
    with _LOCK:
        kept = [*history(p), entry][-MAX_APPLIED:]
        cache.save_json(_file(p, HISTORY), {"applied": kept})


def find_proposal(p: Path, pid: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(turn, proposal)`` of the saved proposal ``pid``, or None."""
    for turn in reversed(turns(p)):
        for event in turn.get("events") or []:
            if isinstance(event, dict) and event.get("kind") == "proposal":
                if event.get("id") == pid:
                    return turn, event
                for sug in event.get("suggestions") or []:
                    if isinstance(sug, dict) and sug.get("id") == pid:
                        return turn, sug
    return None
