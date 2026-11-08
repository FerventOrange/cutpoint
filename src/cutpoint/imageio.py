"""Picture files read and written through bytes.

``cv2.imwrite`` and ``cv2.imread`` fail without raising on a non-ASCII path on Windows
(the picture is just missing), so every picture goes through ``imencode``/``imdecode``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_jpeg(path: Path, img: np.ndarray, quality: int) -> bool:
    """Write ``img`` as a JPEG; through bytes, as ``cv2.imwrite`` fails silently on a
    non-ASCII path on Windows. False when it could not be encoded."""
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        path.write_bytes(buf.tobytes())
    return bool(ok)


def write_png(path: Path, img: np.ndarray) -> bool:
    """Write ``img`` as a PNG, like :func:`write_jpeg`. False when it could not be encoded."""
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".png", img)
    if ok:
        path.write_bytes(buf.tobytes())
    return bool(ok)


def read_image(path: Path) -> np.ndarray | None:
    """The picture at ``path``, or None; read through bytes (see :func:`write_jpeg`)."""
    import cv2  # noqa: PLC0415

    try:
        data = path.read_bytes()
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
