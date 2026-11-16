"""Picture files read and written through bytes.

``cv2.imwrite`` and ``cv2.imread`` fail without raising on a non-ASCII path on Windows
(the picture is just missing), so every picture goes through ``imencode``/``imdecode``.
"""

from __future__ import annotations

import os
import uuid
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np

from cutpoint import cache


def _write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to a temp file beside ``path``, then rename it, so an interrupted write
    never leaves a half or empty picture."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_bytes(data)
        cache.retry_locked(partial(os.replace, tmp, path), path, "replace")
    except BaseException:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def write_jpeg(path: Path, img: np.ndarray, quality: int) -> bool:
    """Write ``img`` as a JPEG; through bytes, as ``cv2.imwrite`` fails silently on a
    non-ASCII path on Windows. False when it could not be encoded."""
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        _write_bytes(path, buf.tobytes())
    return bool(ok)


def write_png(path: Path, img: np.ndarray) -> bool:
    """Write ``img`` as a PNG, like :func:`write_jpeg`. False when it could not be encoded."""
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".png", img)
    if ok:
        _write_bytes(path, buf.tobytes())
    return bool(ok)


def read_image(path: Path) -> np.ndarray | None:
    """The picture at ``path``, or None; read through bytes (see :func:`write_jpeg`)."""
    import cv2  # noqa: PLC0415

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:  # an interrupted write leaves an empty file
        return None
    try:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
