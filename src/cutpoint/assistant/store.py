"""The assistant's files in ``<project>/ui/assistant``.

``chat.json``
    ``{"turns": [...]}``: the finished turns, oldest first (:meth:`cutpoint.assistant.session.Turn.record`).
    Clear chat empties it.
``history.json``
    ``{"applied": [...]}``: each proposal the user applied, with the sequence before and after,
    so it can be reverted after a reload. Clear chat keeps it.

Writes are atomic (:func:`cutpoint.cache.save_json`); a file a link takes outside the project is
never read or written.

Reads come from memory while the file is unchanged (:func:`_load`; #1085 A762): its
modification time, size and inode (:func:`cutpoint.utils.file_stamp`) say when another process
or a write here replaced it. Every change is still written to the file at once, so a reload or a
crash loses nothing; a write only forgets the copy in memory, and the next read parses the new
file. The lists and dicts read are shared with that copy: read them, never change them.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from cutpoint import cache, state
from cutpoint.projectfiles import UI_DIR
from cutpoint.utils import file_stamp

FOLDER = f"{UI_DIR}/assistant"
CHAT = "chat.json"
HISTORY = "history.json"
MAX_TURNS = 200  # the oldest go first
MAX_APPLIED = 500
MAX_CACHED = 16  # files kept parsed in memory (two per project)

_LOCK = threading.Lock()  # one read-modify-write at a time (turn threads and requests)
# path -> (file_stamp, parsed JSON) of the files last read, most recently used last
_PARSED: OrderedDict[str, tuple[tuple[int, int, int], Any]] = state.register_container(
    __name__, "_PARSED", OrderedDict()
)
_PARSED_LOCK = threading.Lock()  # guards _PARSED; never held while reading a file


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


def _load(f: Path) -> Any:
    """:func:`cutpoint.cache.load_json` of ``f``, from memory while the file is the one parsed
    last (same stamp before and after that read); a file held by
    :func:`cutpoint.cache.hold_outputs` is always read."""
    if cache.is_held(f):
        return cache.load_json(f)
    key, before = str(f), file_stamp(f)
    with _PARSED_LOCK:
        held = _PARSED.get(key)
        if held is not None and before is not None and held[0] == before:
            _PARSED.move_to_end(key)
            return held[1]
    data = cache.load_json(f)
    with _PARSED_LOCK:
        if before is not None and file_stamp(f) == before:  # not replaced while it was read
            _PARSED[key] = (before, data)
            _PARSED.move_to_end(key)
            while len(_PARSED) > MAX_CACHED:
                _PARSED.popitem(last=False)
        else:
            _PARSED.pop(key, None)
    return data


def _save(f: Path, data: Any) -> None:
    """Write ``f`` as before (atomic, at once) and forget its parsed copy."""
    try:
        cache.save_json(f, data)
    finally:
        with _PARSED_LOCK:
            _PARSED.pop(str(f), None)


def _items(p: Path, name: str, key: str) -> list[dict[str, Any]]:
    try:
        data = _load(_file(p, name))
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
        _save(_file(p, CHAT), {"turns": kept})


def clear_chat(p: Path) -> None:
    """Forget the chat; the history of applied proposals stays."""
    with _LOCK:
        _save(_file(p, CHAT), {"turns": []})


def history(p: Path) -> list[dict[str, Any]]:
    """The applied proposals, oldest first."""
    return _items(p, HISTORY, "applied")


def add_applied(p: Path, entry: dict[str, Any]) -> None:
    with _LOCK:
        kept = [*history(p), entry][-MAX_APPLIED:]
        _save(_file(p, HISTORY), {"applied": kept})


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
