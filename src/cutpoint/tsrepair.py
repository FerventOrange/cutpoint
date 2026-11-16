"""Joined MPEG-TS and MPEG-PS recordings whose timestamps jump (#1052).

A recording joined from several (OBS, a camera or a capture card restarting its clock, or
files concatenated byte by byte) can have timestamps that jump back to zero, or far ahead, at
each join. ffprobe then reports the time from the first to the last timestamp, so the file
looks shorter (or longer) than it plays, while ffmpeg corrects the jumps in every decode: the
analysis WAV and its transcript run on ffmpeg's clock, but the analysis would stop at the
probed length and a seek past a join would find nothing.

Such a source gets a stream copy (video and audio, no re-encode) remuxed into its cache folder
as ``timeline.mkv``, on ffmpeg's clock; :func:`cutpoint.stage.staged_path` then reads that copy
instead of the recording (the analysis and the clips alike), while every project file keeps the
original path. The verdict of the one stream-copy read that finds the jumps is cached per
source as ``timeline.json``; the copies are listed in ``<cache>/timeline-index.json`` and kept
until the cache is cleared.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Collection, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from cutpoint import cache, folders, jsonout, procs, stage, state
from cutpoint.config import remap_source
from cutpoint.errors import FFmpegError, FolderBusyError
from cutpoint.ffmpeg_utils import MediaInfo, probe, run_progress

logger = logging.getLogger(__name__)

VERSION = 1
TIMELINE_FILE = "timeline.json"  # in the source's cache folder: the verdict
COPY_FILE = "timeline.mkv"  # in the source's cache folder: the stream copy
# Containers whose probed length libavformat takes from the first and last timestamp
# (ffprobe's format_name, first name)
CONTAINERS = ("mpegts", "mpeg")
MISMATCH_S = 1.0  # read and probed length this far apart (with a jump) make a copy
FREE_FACTOR = 1.05  # free space the copy needs, times the source's size
_DISCONTINUITY = re.compile(r"timestamp discontinuity[^\n]*")
_PART = re.compile(r"timeline\.(\d+)\.[0-9a-f]{8}\.part\.mkv")
# What a source keeps of its cache when it is first marked for a copy: the audio and what is
# made of it come out the same from the copy (ffmpeg put the audio on the container's clock when
# it was extracted, and audio16k-clock.json says so: a WAV without it is checked again, #1056)
_KEPT = ("audio16k.wav", "audio16k-tracks.json", "audio16k-clock.json", "loudness.json",
         "transcript-*.json",
         "fallback-transcript-*.json", TIMELINE_FILE, COPY_FILE, "timeline.*.part.mkv",
         ".timeline.cutpoint-lock")  # fmt: skip

# (stage key, copy size) of the copies this process has logged (`edit` runs the analysis and
# the clips, a `cutpoint ui` many reads: the INFO line comes once)
_REPORTED: set[tuple[str, int]] = state.register_container(__name__, "_REPORTED", set())
# stage key -> the message of the copy that failed this run: not tried again (a copy that
# appears halfway would drop the caches this run wrote), the warning code still reported
_FAILED: dict[str, str] = state.register_container(__name__, "_FAILED", {})


@dataclass(frozen=True)
class Verdict:
    """What one stream-copy read of a source found.

    Attributes:
        discontinuities: Timestamp jumps ffmpeg corrected.
        probed_s: Length ffprobe reports (s).
        read_s: Length ffmpeg reads (s), on its corrected clock.
        remux: Whether the source is read through a stream copy.
    """

    discontinuities: int
    probed_s: float
    read_s: float
    remux: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "discontinuities": self.discontinuities,
            "probed_s": round(self.probed_s, 3),
            "read_s": round(self.read_s, 3),
            "remux": self.remux,
        }

    @classmethod
    def from_json(cls, data: Any) -> Verdict | None:
        if not isinstance(data, dict) or data.get("version") != VERSION:
            return None
        try:
            return cls(
                int(data["discontinuities"]),
                float(data["probed_s"]),
                float(data["read_s"]),
                bool(data["remux"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def record(self) -> dict[str, Any]:
        """The ``timeline`` entry of a source in highlights.json."""
        return {
            "remuxed": True,
            "probed_s": round(self.probed_s, 3),
            "read_s": round(self.read_s, 3),
            "discontinuities": self.discontinuities,
        }


def scanned(info: MediaInfo) -> bool:
    """Whether a source with probe ``info`` is checked for timestamp jumps (MPEG-TS, MPEG-PS)."""
    return info.container.split(",")[0] in CONTAINERS


def check(src: Path, info: MediaInfo | None = None) -> Verdict | None:
    """The verdict for ``src`` (cached in its cache folder), or None when it is not checked:
    a container other than MPEG-TS/PS, or a read that failed.

    ``info``: the probe of the recording itself (not of its copy); probed when None.
    """
    found = Verdict.from_json(cache.load_json(cache.source_dir(src, create=False) / TIMELINE_FILE))
    if found is not None:
        return found
    io = remap_source(src)
    with stage.unrepaired():
        if info is None:
            info = probe(io)
        if not scanned(info):
            return None
        stream = "0:a:0" if info.audio_streams else "0:v:0"
        # -v debug: ffmpeg 5.1 logs the jump only at debug level (6.1 and later at warning);
        # a stream copy logs no line per packet, so this stays short (about 150 lines for 2 h)
        args = ["ffmpeg", "-nostdin", "-v", "debug", "-i", str(io), "-map", stream,
                "-c", "copy", "-f", "null", "-"]  # fmt: skip
        try:
            res = run_progress(args, info.duration, f"timestamp check {src.name}")
        except FFmpegError as err:
            logger.info("%s: timestamp check failed (%s); read as it is", src.name, err)
            return None
    jumps = len(_DISCONTINUITY.findall(res.stderr))
    remux = jumps >= 1 and abs(res.out_time - info.duration) >= MISMATCH_S
    verdict = Verdict(jumps, info.duration, res.out_time, remux)
    cache.save_json(cache.source_dir(src) / TIMELINE_FILE, verdict.to_json())
    return verdict


def _drop_stale_caches(cdir: Path) -> None:
    """Remove what a source's cache folder holds from reads of the recording cut short at the
    probed length (scene cuts, read errors, events, OCR, vision, category), when a copy is made
    after reads of the recording itself; the audio, loudness and transcripts are kept."""
    for entry in cdir.iterdir():
        if any(entry.match(pattern) for pattern in _KEPT):
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError as err:
            logger.debug("could not remove stale cache %s: %s", entry, err)


def ensure_copy(src: Path, verdict: Verdict) -> Path | None:
    """The stream copy of ``src`` on ffmpeg's clock, remuxed into its cache folder when it is
    missing; None when it cannot be made (too little free space, ffmpeg failed: a WARNING and
    the ``--json`` warning ``timeline_remux_failed``; not tried again in this process).

    One run at a time makes the copy of a source (a lock in its cache folder); another waits
    and uses the copy that run made.
    """
    io = remap_source(src)
    st = io.stat()
    cdir = cache.source_dir(src)
    copy = cdir / COPY_FILE
    key = stage.stage_key(io)
    if key in _FAILED:
        if not jsonout.has_warning("timeline_remux_failed", [src]):
            jsonout.warn("timeline_remux_failed", _FAILED[key], [src])
        return None
    if _registered(key, copy, st):
        return copy
    with _copy_lock(cdir, src):
        return _make_copy(src, verdict, io, st, cdir, key)


@contextlib.contextmanager
def _copy_lock(cdir: Path, src: Path) -> Iterator[None]:
    """Hold the lock of the copy in ``cdir`` (:func:`cutpoint.folders.swap_lock`), waiting for
    another run that holds it (a stale one is taken over)."""
    held = ExitStack()
    waiting = False
    while True:
        try:
            held.enter_context(folders.swap_lock(cdir / "timeline"))
            break
        except FolderBusyError:
            if not waiting:
                waiting = True
                logger.info("%s: waiting for another run to finish its stream copy", src.name)
            time.sleep(0.2)
    with held:
        yield


def _make_copy(
    src: Path, verdict: Verdict, io: Path, st: os.stat_result, cdir: Path, key: str
) -> Path | None:
    """:func:`ensure_copy` holding the copy's lock: the copy another run made while this one
    waited is used, else it is remuxed."""
    copy = cdir / COPY_FILE
    if _registered(key, copy, st):
        return copy
    stored = cache.load_json(cdir / TIMELINE_FILE) or {}
    if copy.is_file() and stored.get("copy_size") == copy.stat().st_size:
        _register(key, copy, st)  # the index lost it (another run wrote it meanwhile)
        return copy
    _clean_parts(cdir)
    free = shutil.disk_usage(cdir).free
    if free < FREE_FACTOR * st.st_size:
        _failed(
            src,
            verdict,
            f"too little free space in {cdir.parent} ({_size(free)} free, "
            f"{_size(round(FREE_FACTOR * st.st_size))} needed)",
            key,
        )
        return None
    part = cdir / f"timeline.{os.getpid()}.{uuid.uuid4().hex[:8]}.part.mkv"
    logger.info("%s: remuxing a stream copy (%s) into the cache", src.name, _size(st.st_size))
    args = ["ffmpeg", "-nostdin", "-y", "-v", "warning", "-i", str(io), "-map", "0:v",
            "-map", "0:a?", "-c", "copy", "-ignore_unknown", "-f", "matroska", str(part)]  # fmt: skip
    try:
        with stage.unrepaired():
            run_progress(args, verdict.read_s, f"timeline remux {src.name}")
        # os.replace, not Path.replace: tests patch cache.os.replace
        cache.retry_locked(partial(os.replace, part, copy), copy, "replace")
    except FFmpegError as err:
        with contextlib.suppress(OSError):
            part.unlink(missing_ok=True)
        _failed(src, verdict, f"ffmpeg failed: {err}", key)
        return None
    except BaseException:
        with contextlib.suppress(OSError):  # still locked (Windows, right after Ctrl+C)
            part.unlink(missing_ok=True)
        raise
    if "copy_size" not in stored:  # the reads so far were of the recording, cut short
        _drop_stale_caches(cdir)
    size = copy.stat().st_size
    cache.save_json(cdir / TIMELINE_FILE, {**verdict.to_json(), "copy_size": size})
    _register(key, copy, st)
    return copy


def _registered(key: str, copy: Path, st: os.stat_result) -> bool:
    """Whether the timeline index lists ``copy`` for the source ``key`` as it is now."""
    entry = stage.timeline_index().get(key)
    if not isinstance(entry, dict):
        return False
    try:
        return (
            Path(cache.cache_root() / entry["file"]) == copy
            and entry["size"] == st.st_size
            and entry["mtime_ns"] == st.st_mtime_ns
            and copy.stat().st_size == entry["copy_size"]
        )
    except (OSError, KeyError, TypeError):
        return False


def _register(key: str, copy: Path, st: os.stat_result) -> None:
    """List ``copy`` in the timeline index for the source ``key``."""
    root = cache.cache_root()
    path = root / stage.TIMELINE_INDEX
    data = cache.load_json(path)
    index = data if isinstance(data, dict) else {}
    index[key] = {
        "file": copy.relative_to(root).as_posix(),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "copy_size": copy.stat().st_size,
    }
    cache.save_json(path, index)


def _clean_parts(cdir: Path) -> None:
    """Remove the ``.part`` copies of runs that are gone (a hard-killed run leaves its own)."""
    for part in cdir.glob("timeline.*.part.mkv"):
        m = _PART.fullmatch(part.name)
        try:
            if m and procs.pid_alive(int(m.group(1)), part.stat().st_mtime):
                continue
            part.unlink()
        except OSError as err:
            logger.debug("could not remove %s: %s", part, err)


def _size(n: int) -> str:
    """``5.6 GB``, or ``120 MB`` below 1 GB."""
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _jumps(verdict: Verdict) -> str:
    """``its recordings' timestamps jump back at 1 join (the file says 12.0 s, it plays
    24.0 s)``."""
    way = "back" if verdict.read_s > verdict.probed_s else "ahead"
    joins = "join" if verdict.discontinuities == 1 else "joins"
    return (
        f"its recordings' timestamps jump {way} at {verdict.discontinuities} {joins} "
        f"(the file says {verdict.probed_s:.1f} s, it plays {verdict.read_s:.1f} s)"
    )


def _failed(src: Path, verdict: Verdict, why: str, key: str) -> None:
    message = (
        f"{src.name}: {_jumps(verdict)}, but no stream copy could be made ({why}); it is "
        "read as it is, so the footage after the first join may be left out"
    )
    logger.warning("%s", message)
    _FAILED[key] = message
    if not jsonout.has_warning("timeline_remux_failed", [src]):
        jsonout.warn("timeline_remux_failed", message, [src])


def _no_copy(src: Path, why: str, then: str = "re-run once the copy can be made") -> FFmpegError:
    """The error for a source whose recorded times are on its stream copy's timeline, read
    without the copy."""
    return FFmpegError(
        f"{src.name}: its times are on the timeline of its stream copy, but {why}; {then}"
    )


def prepare_sources(
    sources: Iterable[Path], *, required: Collection[str] = ()
) -> dict[str, dict[str, Any]]:
    """Check every source and make the stream copies the ones with timestamp jumps need.

    Returns:
        The highlights.json ``timeline`` entry (:meth:`Verdict.record`) by source path, for
        the sources read through a copy.

    Raises:
        FFmpegError: When a source in ``required`` (paths whose recorded times are on the
            copy's clock, such as a project's remuxed sources) gets no copy (its check failed,
            finds no jumps or the copy could not be made): read in place, its times would
            point elsewhere. A missing file is left to the caller's read.
    """
    out: dict[str, dict[str, Any]] = {}
    needed = {stage.stage_key(p) for p in required}
    for src in dict.fromkeys(sources):
        need = stage.stage_key(src) in needed
        try:
            verdict = check(src)
        except (OSError, FFmpegError) as err:  # missing or unreadable: the caller's read says so
            if need and remap_source(src).exists():
                raise _no_copy(src, f"its timestamp check failed: {err}") from err
            logger.debug("%s: timestamp check skipped (%s)", src, err)
            continue
        if verdict is None or not verdict.remux:
            if need:
                why = (
                    "its timestamp check gave no verdict"
                    if verdict is None
                    else "no jumps were found"
                )
                raise _no_copy(src, f"{why} this time", "analyze the project again")
            continue
        copy = ensure_copy(src, verdict)
        if copy is None:
            if need:
                raise _no_copy(src, "it could not be made (see the WARNING above)")
            # read as it is from here: the next copy drops what these reads cache
            with contextlib.suppress(OSError):
                cache.save_json(cache.source_dir(src) / TIMELINE_FILE, verdict.to_json())
            continue
        out[str(src)] = verdict.record()
        size = copy.stat().st_size
        message = (
            f"{src.name}: {_jumps(verdict)}; reading a stream copy remuxed into the cache "
            f"({_size(size)})"
        )
        seen = (stage.stage_key(src), size)
        if seen not in _REPORTED:
            _REPORTED.add(seen)
            logger.info("%s", message)
        if not jsonout.has_warning("timeline_remuxed", [src]):
            jsonout.warn("timeline_remuxed", message, [src])
    return out
