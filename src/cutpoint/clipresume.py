"""Finished clips of an interrupted ``clips`` or ``extend`` export, taken over by the next run.

A staged export (:func:`cutpoint.folders.staged_dir`) writes ``resume.json`` into its staging
folder after each finished clip: per clip file name the :func:`clip_key` of what it rendered,
its size and its read-error record. When the run fails, is cancelled or is killed hard, the
staging folder is kept as the hidden ``.<folder>.cutpoint-resume`` next to the clips folder
(:func:`cutpoint.folders.resume_path`). The next export into that folder moves the clips whose
name, key and size match its own plan into its staging folder and encodes only the rest; the
resume folder is then removed, so a run with another plan deletes it. 360 clips and clips
trimmed by a read error are never taken over. Nothing in the resume folder is ever visible in the clips folder itself.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from cutpoint import __version__, cache, folders
from cutpoint.encode import Grade, Lut, OutputFormat

logger = logging.getLogger(__name__)

VERSION = 1


def clip_key(
    src: Path,
    start: float,
    end: float,
    fmt: OutputFormat,
    vertical: bool,
    crop: Any,
    audio_fade: float,
    peak: str,
    grade: Grade | None,
) -> str:
    """Identity of one clip render (the arguments of
    :func:`~cutpoint.encode.export_clip_checked`, the size and modification time of the source
    and of a LUT file, and the Cutpoint version): an earlier clip with the same key renders
    exactly like the new one."""
    return cache.params_hash(
        {
            "version": __version__,
            "src": str(src),
            "stamp": _stamp(src),
            "lut_stamp": _stamp(grade.path) if isinstance(grade, Lut) else None,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "format": [fmt.width, fmt.height, fmt.fps, bool(vertical)],
            "crop": None if crop is None else str(crop),
            "audio_fade": float(audio_fade),
            "peak": peak,
            "grade": grade.to_json() if grade is not None else None,
        },
        length=16,
    )


def _stamp(path: Path) -> list[int] | None:
    try:
        st = os.stat(path)  # noqa: PTH116 - one call for size and mtime
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


class ClipResume:
    """``resume.json`` of one staged export, and the clips taken over from the last one.

    Args:
        dst: The clips folder exported into.
        work: Its staging folder.
        command: ``clips`` or ``extend`` (recorded only).
        keys: :func:`clip_key` per clip file name of this run's plan (360 clips left out).
    """

    def __init__(self, dst: Path, work: Path, command: str, keys: dict[str, str]) -> None:
        self.path = work / folders.RESUME_FILE
        self.keys = keys
        self._lock = threading.Lock()  # export_workers finish clips in threads
        self.data: dict[str, Any] = {
            "version": VERSION,
            "generator": f"cutpoint {__version__}",
            "command": command,
            "clips": {},
        }
        self.taken: dict[str, dict[str, Any] | None] = {}  # name: its read-error record
        self._take_over(dst, work)

    def _take_over(self, dst: Path, work: Path) -> None:
        folder = folders.resume_path(dst)
        if not folder.is_dir():
            return
        rec = cache.load_json(folder / folders.RESUME_FILE)
        done = rec.get("clips") if isinstance(rec, dict) and rec.get("version") == VERSION else {}
        for name, key in self.keys.items():
            c = done.get(name) if isinstance(done, dict) else None
            f = folder / name
            # the clip as it was recorded finished: a partial one is never listed
            # as extend's _reusable: a clip trimmed by a read error is exported again
            if not (isinstance(c, dict) and c.get("key") == key and f.is_file()):
                continue
            if c.get("read_error"):
                continue
            try:
                if f.stat().st_size != c.get("size"):
                    continue
                f.replace(work / name)
            except OSError as err:  # locked (Windows): exported again
                logger.debug("Not taking over %s: %s", f, err)
                continue
            self.taken[name] = None
            self.add(name, None)
        if self.taken:
            logger.info(
                "Took over %d finished clip(s) of an interrupted run from %s; exporting the "
                "other %d",
                len(self.taken),
                folder,
                len(self.keys) - len(self.taken),
            )
        else:
            logger.info("Removing %s: its clips are not part of this run", folder)
        try:
            folders._remove_leftover(
                folder, f"the finished clips of an interrupted export into {dst}"
            )
        except OSError as err:  # still open (a player, a virus scanner)
            logger.warning("Could not remove %s (%s); the next run removes it", folder, err)

    def add(self, name: str, read_error: dict[str, Any] | None) -> None:
        """Record the finished clip ``name`` (no-op for a clip without a key)."""
        key = self.keys.get(name)
        if key is None:
            return
        size = (self.path.parent / name).stat().st_size
        with self._lock:
            self.data["clips"][name] = {"key": key, "size": size, "read_error": read_error}
            cache.save_json(self.path, self.data)

    def discard(self) -> None:
        """Every clip is exported: drop ``resume.json`` before the clips are swapped in."""
        with self._lock:
            self.path.unlink(missing_ok=True)
