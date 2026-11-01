"""The synthetic HUD clip of the discovery end-to-end test (OpenCV text, invented game) and
the OCR cache comparison, shared by the OCR tests."""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint.events import ocr_raw

W, H, FPS, SECONDS = 1280, 720, 4, 60
GAME = "tilefall"  # invented game key
NAMES = ["Ash", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren", "Brim"]
# drawn feed rectangle (x, y, w, h) in the 1280x720 frame: top right
FEED = (840, 30, 420, 130)


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    """Draw ``text`` with its baseline at ``org`` on a dark box, so OCR reads it cleanly."""
    import cv2  # noqa: PLC0415

    thick = 2 if scale < 1.1 else 3
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + base + 8), (10, 10, 10), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _frame(base: np.ndarray, t: float, seconds: int) -> np.ndarray:
    """One frame at time ``t``: the static watermark, the feed and the notice."""
    img = base.copy()
    slot = int(t / 1.2)
    if t < (slot + 1) * 1.2 - 0.05:  # feed text changes every 1.2 s, with a short gap
        for row in range(3):
            text = f"{NAMES[(slot + row) % 10]} downed {NAMES[(slot * 3 + row * 7 + 1) % 10]}"
            _label(img, text, (860, 70 + row * 36), 0.9, (255, 255, 255))
    k = int(t // 6)
    if k < seconds // 6 and k * 6 + 1.0 <= t <= k * 6 + 3.5:  # notice: 2.5 s of every 6 s
        text = f"Gate {NAMES[k % 10]} opened"
        import cv2  # noqa: PLC0415

        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        _label(img, text, ((W - tw) // 2, 360), 1.3, (0, 255, 255))
    return img


def make_clip(path: Path, seconds: int = SECONDS) -> None:
    """Write the synthetic clip (``seconds`` long): frames drawn with OpenCV, piped as raw bgr24
    into ffmpeg."""
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = (0x40, 0x30, 0x20)  # bgr of 0x203040
    _label(base, "MARKER 07", (20, 690), 0.9, (255, 255, 255))  # static watermark
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(path),
    ]  # fmt: skip
    log = path.with_suffix(".log")  # stderr to a file: a full pipe cannot block ffmpeg
    with log.open("wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        assert proc.stdin is not None
        try:
            for i in range(seconds * FPS):
                proc.stdin.write(_frame(base, i / FPS, seconds).tobytes())
        except OSError:  # BrokenPipeError, or EINVAL on Windows
            pass  # ffmpeg exited early; the return code below reports why
        finally:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        try:
            rc = proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
    assert rc == 0, log.read_text(errors="replace")[-300:]


def ocr_cache_files(folder: Path) -> dict[str, Any]:
    """Every cache file under ``folder`` by relative path: the bytes, a raw chunk's decoded
    content (its zip and gzip headers hold the write time) and the OCR result without its
    ``elapsed_s`` and source path (each run has its own folder)."""
    out: dict[str, Any] = {}
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        data: Any = p.read_bytes()
        if p.name.startswith("events-ocr-"):
            doc = json.loads(data)
            doc["stats"].pop("elapsed_s")
            doc.pop("source")
            data = doc
        elif p.suffix == ".zip":
            chunk = ocr_raw._decode(data)
            data = (
                chunk.frames,
                chunk.state,
                {k: v.tobytes() for k, v in chunk.images.items()},
                {k: v.tobytes() for k, v in chunk.arrays.items()},
            )
        out[p.relative_to(folder).as_posix()] = data
    return out
