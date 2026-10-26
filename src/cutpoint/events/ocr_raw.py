"""Raw OCR cache: what the OCR layer's reader saw on every frame of a source.

:func:`~cutpoint.events.killfeed.detect_ocr_events` splits into a reader (the
gates and the OCR calls) and the interpretation. The reader's records go here,
so a change of the interpretation (a ``GENERIC_VERSION`` bump) replays them in
seconds, without one OCR call, instead of reading the footage again.

Layout, in the source's cache folder: ``ocr-raw-<hash of raw_key>/`` with
``meta.json`` (the key, the finished chunks, whether the read is complete) and
``chunk-00000.zip`` ... of about :data:`CHUNK_FRAMES` frames each. A chunk holds
``frames.json.gz`` (the frame records and the reader's state at the chunk's
end), the PNG bands of the rows read in feed regions and ``state.npz`` (the
still thumbnails and the reader's masks). Chunks are written atomically and
``meta.json`` last, so an interrupted read resumes after its last chunk.

The records hold OCR text, so the folder stays in the cache (git-ignored) and is
never written inside the package's git checkout.
"""

from __future__ import annotations

import contextlib
import gzip
import importlib.metadata
import io
import json
import logging
import os
import shutil
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache

logger = logging.getLogger(__name__)

# bump when the reader's records change (what is recorded, or how a gate decides)
RAW_VERSION = 1
CHUNK_FRAMES = 600  # frames per chunk: what an interrupted read loses at most
KEEP_RAW = 2  # complete raw reads kept per source (the most recently used)
STALE_S = 7 * 24 * 3600  # an incomplete raw read untouched this long is removed


def engine_version() -> str | None:
    """Version of the installed RapidOCR package (None when it is missing)."""
    try:
        return importlib.metadata.version("rapidocr-onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return None


def raw_key(
    fps: float,
    regions: list[dict[str, Any]],
    newest: str,
    min_text_score: float,
    avoid: list[tuple[float, float]],
    det_side: int,
) -> dict[str, Any]:
    """Key of a raw read: everything the reader's records depend on.

    Args:
        fps: OCR frame rate.
        regions: :meth:`~cutpoint.events.regions.Region.key` of every region.
        newest: The feed end new entries appear at (the preset's ``feed.newest``).
        min_text_score: Recognition score below which text is dropped.
        avoid: Time ranges never decoded.
        det_side: Detection size limit of the OCR engine.
    """
    return {
        "v": RAW_VERSION,
        "fps": fps,
        "regions": regions,
        "newest": newest,
        "min_text_score": min_text_score,
        "avoid": [list(a) for a in avoid],
        "engine": engine_version(),
        "det_side": det_side,
    }


@dataclass
class Chunk:
    """One chunk of a raw read: JSON frame records, the reader's state at its end
    (JSON), images stored as PNG (uint8, lossless) and other arrays (npz)."""

    frames: list[dict[str, Any]]
    state: dict[str, Any]
    images: dict[str, np.ndarray] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def _checkout(here: Path = Path(__file__)) -> Path | None:
    """Cutpoint's own git checkout the package runs from (``<checkout>/src/cutpoint/
    events/ocr_raw.py``), if any; a git repository merely around the install (a home
    folder kept in git) is none."""
    parents = here.resolve().parents
    if len(parents) > 3 and parents[2].name == "src" and (parents[3] / ".git").exists():
        return parents[3]
    return None


def in_checkout(folder: Path) -> bool:
    """Whether ``folder`` resolves inside the package's git checkout."""
    root = _checkout()
    return root is not None and folder.resolve().is_relative_to(root)


def _encode(chunk: Chunk) -> bytes:
    import cv2  # noqa: PLC0415

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        blob = json.dumps({"frames": chunk.frames, "state": chunk.state}, ensure_ascii=False)
        zf.writestr("frames.json.gz", gzip.compress(blob.encode("utf-8"), compresslevel=6))
        for name, img in chunk.images.items():
            ok, png = cv2.imencode(".png", np.ascontiguousarray(img))
            if not ok:
                raise OSError(f"cannot encode {name}.png")
            zf.writestr(f"{name}.png", png.tobytes())
        arrays = io.BytesIO()
        np.savez_compressed(arrays, **chunk.arrays)
        zf.writestr("state.npz", arrays.getvalue())
    return buf.getvalue()


def _decode(data: bytes) -> Chunk:
    import cv2  # noqa: PLC0415

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        blob = json.loads(gzip.decompress(zf.read("frames.json.gz")).decode("utf-8"))
        images = {}
        for name in zf.namelist():
            if name.endswith(".png"):
                buf = np.frombuffer(zf.read(name), np.uint8)
                images[name[:-4]] = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        with np.load(io.BytesIO(zf.read("state.npz"))) as npz:
            arrays = {k: npz[k] for k in npz.files}
    return Chunk(blob["frames"], blob["state"], images, arrays)


class RawStore:
    """The raw read of one source for one :func:`raw_key`."""

    def __init__(self, cdir: Path, key: dict[str, Any]) -> None:
        self.cdir = cdir
        self.key = json.loads(json.dumps(key))  # as read back from meta.json
        self.folder = cdir / f"ocr-raw-{cache.params_hash(key)}"
        self.meta_path = self.folder / "meta.json"
        self._meta: dict[str, Any] = {}

    def chunk_path(self, i: int) -> Path:
        return self.folder / f"chunk-{i:05d}.zip"

    def meta(self) -> dict[str, Any] | None:
        """``meta.json`` (``key``, ``chunks``, ``complete``, ``info``, ``stats``), or
        None when there is none for this key."""
        meta = cache.load_json(self.meta_path)
        if not isinstance(meta, dict) or meta.get("key") != self.key:
            return None
        return meta

    def chunks(self, meta: dict[str, Any]) -> Iterator[Chunk]:
        """The finished chunks in order, up to the first unreadable one."""
        for i in range(int(meta.get("chunks", 0))):
            try:
                chunk = _decode(self.chunk_path(i).read_bytes())
            except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as err:
                logger.warning("%s: raw OCR chunk %d unreadable (%s)", self.folder.name, i, err)
                return
            yield chunk

    def touch(self) -> None:
        """Mark the read as used now (cleanup keeps the most recently used)."""
        with contextlib.suppress(OSError):
            os.utime(self.meta_path)

    def drop(self) -> None:
        """Delete the raw read."""
        shutil.rmtree(self.folder, ignore_errors=True)

    def writable(self) -> bool:
        """Whether the read may be written here: never inside the package's git
        checkout (the records hold OCR text). Warns when it may not."""
        if in_checkout(self.folder):
            logger.warning(
                "%s is inside the cutpoint git checkout; not writing the raw OCR cache there "
                "(set CUTPOINT_HOME or CUTPOINT_CACHE_DIR to a folder outside it)",
                self.folder,
            )
            return False
        return True

    @property
    def started(self) -> bool:
        """Whether :meth:`start` ran."""
        return bool(self._meta)

    def start(self, done: int, info: dict[str, Any]) -> None:
        """Begin (or, after ``done`` finished chunks, continue) writing: chunks from
        ``done`` on are removed and ``meta.json`` says the read is incomplete."""
        self.folder.mkdir(parents=True, exist_ok=True)
        for p in self.folder.glob("chunk-*.zip"):
            try:
                stale = int(p.stem.split("-")[1]) >= done
            except (IndexError, ValueError):
                stale = True
            if stale:
                p.unlink(missing_ok=True)
        self._meta = {"key": self.key, "chunks": done, "complete": False, "info": info}
        cache.save_json(self.meta_path, self._meta)

    def add(self, chunk: Chunk) -> None:
        """Write the next chunk, then ``meta.json``."""
        n = self._meta["chunks"]
        cache.save_bytes(self.chunk_path(n), _encode(chunk))
        self._meta["chunks"] = n + 1
        cache.save_json(self.meta_path, self._meta)

    def fail(self) -> None:
        """Mark the read failed (a part gave no frames): it stays incomplete and the next
        read starts over instead of resuming after the gap."""
        self._meta["failed"] = True
        cache.save_json(self.meta_path, self._meta)

    def finish(self, stats: dict[str, Any]) -> None:
        """Mark the read complete (with the reader's ``stats``) and clean up the
        source's other raw reads (:func:`cleanup`)."""
        self._meta |= {"complete": True, "stats": stats}
        cache.save_json(self.meta_path, self._meta)
        cleanup(self.cdir, self.folder)


def cleanup(cdir: Path, current: Path, keep: int = KEEP_RAW, stale_s: float = STALE_S) -> None:
    """Keep ``current`` and the most recently used other complete raw reads of a
    source, ``keep`` in all; remove incomplete ones untouched for ``stale_s``."""
    now = time.time()
    complete: list[tuple[float, Path]] = []
    for folder in cdir.glob("ocr-raw-*"):
        if not folder.is_dir() or folder == current:
            continue
        meta_path = folder / "meta.json"
        try:
            used = (meta_path if meta_path.exists() else folder).stat().st_mtime
        except OSError:
            continue
        meta = cache.load_json(meta_path)
        if isinstance(meta, dict) and meta.get("complete"):
            complete.append((used, folder))
        elif now - used > stale_s:
            shutil.rmtree(folder, ignore_errors=True)
    complete.sort(reverse=True)
    for _, folder in complete[max(0, keep - 1) :]:
        shutil.rmtree(folder, ignore_errors=True)
