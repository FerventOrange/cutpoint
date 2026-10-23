"""Per-source-file cache (audio, loudness, transcripts, scene cuts)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from cutpoint import paths


def cache_root() -> Path:
    """Cache directory (see :func:`cutpoint.paths.cache_dir`); created if missing."""
    root = paths.cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_dir(src: Path, offset: float = 0.0) -> Path:
    """Cache folder for one source, keyed by absolute path + size + mtime.

    The path in the key is canonicalised through ``source_remap`` (see
    :mod:`cutpoint.config`), so a remapped copy keys like its original.

    Args:
        src: Source video.
        offset: Start of a virtual source (``analyze --recover`` resumes a
            damaged file here); 0 for the file itself. Each offset gets its own
            folder, so the cached audio/transcript of a part starts at 0.
    """
    from cutpoint.config import canonical_source, remap_source  # noqa: PLC0415

    # Stat the file actually read (a remapped copy at the footage's new location), but key
    # it by its original path so an identical copy (same size + mtime) keeps hits.
    io_path = remap_source(src)
    st = io_path.stat()
    key = f"{canonical_source(io_path.resolve())}|{st.st_size}|{st.st_mtime_ns}"
    tag = ""
    if offset > 0:
        key += f"|offset={offset:.3f}"
        tag = f"@{offset:.0f}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    folder = cache_root() / f"{src.stem[:40]}{tag}-{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_json(path: Path) -> Any | None:
    """Load JSON from ``path`` or return None if missing/corrupt (a UTF-8 BOM is allowed)."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):  # ValueError: JSONDecodeError, UnicodeDecodeError
        return None


REPLACE_ATTEMPTS = 5


def _umask() -> int:
    """The process umask.

    Linux: the ``Umask:`` line of ``/proc/self/status``. Elsewhere (or without
    it) by setting it and restoring it at once.
    """
    with contextlib.suppress(OSError, ValueError):
        for line in (
            Path("/proc/self/status").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    mask = os.umask(0)
    os.umask(mask)
    return mask


def match_mode(tmp: Path, target: Path) -> None:
    """Give a ``tempfile`` file (created 0600) the mode a plain write of
    ``target`` would get, before it is renamed onto ``target``.

    POSIX only: the existing target's mode if there is one, else ``0o666``
    minus the umask. No-op on Windows.
    """
    if os.name != "posix":
        return
    try:
        mode = target.stat().st_mode & 0o7777
    except OSError:
        mode = 0o666 & ~_umask()
    with contextlib.suppress(OSError):  # some SMB/CIFS/FUSE mounts reject chmod
        tmp.chmod(mode)


def save_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON (unique temp file, then rename).

    On Windows the rename fails with PermissionError while another process has
    ``path`` open; it is retried a few times with a short backoff
    (:func:`cutpoint.ffmpeg_utils.retry_locked`). The temp file is removed if the
    rename never succeeds.

    Raises:
        PermissionError: If ``path`` stays locked through all attempts (naming it and
            the program that may hold it, see :func:`~cutpoint.ffmpeg_utils.retry_locked`).
    """
    from cutpoint.ffmpeg_utils import retry_locked  # noqa: PLC0415 - it imports this module

    text = json.dumps(data, indent=indent, ensure_ascii=False)
    tmp: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as f:
            tmp = Path(f.name)
            f.write(text)
        match_mode(tmp, path)
        # os.replace, not Path.replace: tests patch cache.os.replace
        retry_locked(partial(os.replace, tmp, path), path, "replace")
        replaced = True
    finally:
        if tmp is not None and not replaced:
            tmp.unlink(missing_ok=True)


def params_hash(params: dict[str, Any], length: int = 10) -> str:
    """Short stable hash of a parameter dict (cache keys and file names).

    ``json.dumps(params, sort_keys=True, default=str)`` hashed with SHA-1; for
    JSON-serialisable dicts these are the same bytes the modules hashed before
    this helper existed, so existing cache entries keep matching.
    """
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]
