"""Real HUD discovery end to end: a synthetic clip drawn with OpenCV (invented HUD-like text)
and encoded by ffmpeg, the real OCR engine, ``pipeline._discover`` writing proposal.json and a
preset.

Skips without ffmpeg or rapidocr (events extra).
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import needs_ffmpeg
from cutpoint import paths
from cutpoint.events import common, pipeline
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.regions import load_preset

W, H, FPS, SECONDS = 1280, 720, 4, 60
GAME = "tilefall"  # invented game key
NAMES = ["Ash", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren", "Brim"]
# drawn feed rectangle (x, y, w, h) in the 1280x720 frame: top right
FEED = (840, 30, 420, 130)


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple) -> None:
    """Draw ``text`` with its baseline at ``org`` on a dark box, so OCR reads it cleanly."""
    thick = 2 if scale < 1.1 else 3
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + base + 8), (10, 10, 10), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _frame(base: np.ndarray, t: float) -> np.ndarray:
    """One frame at time ``t``: the static watermark, the feed and the notice."""
    img = base.copy()
    slot = int(t / 1.2)
    if t < (slot + 1) * 1.2 - 0.05:  # feed text changes every 1.2 s, with a short gap
        for row in range(3):
            text = f"{NAMES[(slot + row) % 10]} downed {NAMES[(slot * 3 + row * 7 + 1) % 10]}"
            _label(img, text, (860, 70 + row * 36), 0.9, (255, 255, 255))
    k = int(t // 6)
    if k < SECONDS // 6 and k * 6 + 1.0 <= t <= k * 6 + 3.5:  # notice: 2.5 s of every 6 s
        text = f"Gate {NAMES[k % 10]} opened"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        _label(img, text, ((W - tw) // 2, 360), 1.3, (0, 255, 255))
    return img


def make_clip(path: Path) -> None:
    """Write the synthetic clip: frames drawn with OpenCV, piped as raw bgr24 into ffmpeg."""
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
            for i in range(SECONDS * FPS):
                proc.stdin.write(_frame(base, i / FPS).tobytes())
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


@needs_ffmpeg
@pytest.mark.slow
def test_discovery_proposes_feed_region_and_installs_preset(home: Path, tmp_path: Path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    clip = tmp_path / "footage" / "session.mp4"
    clip.parent.mkdir()
    make_clip(clip)
    project = tmp_path / "proj"
    project.mkdir()

    # The duration is the one run_events reads from highlights.json; without it load_source
    # falls back to the cached WAV length, and there is none here.
    res = pipeline._discover(
        project, [common.load_source(clip, float(SECONDS))], OcrParams(game=GAME), None, 10, "user"
    )

    assert set(res) == {GAME}  # the key we passed, not a silent fallback to another name
    rec = res[GAME]
    assert rec["installed"] is True
    proposal = json.loads(Path(rec["proposal"]).read_text(encoding="utf-8"))
    assert proposal["game"] == GAME
    assert proposal["regions"], "discovery proposed no region"
    assert (project / "discovery" / GAME / "overlay.jpg").exists()

    # a region overlaps the drawn feed (reference frame is 1920x1080: scale by 1.5)
    fx, fy, fw, fh = (v * 1.5 for v in FEED)
    near = [
        r for r in proposal["regions"]
        if r["x"] < fx + fw and r["x"] + r["w"] > fx and r["y"] < fy + fh and r["y"] + r["h"] > fy
    ]  # fmt: skip
    assert near, f"no region near the feed {FEED}: {proposal['regions']}"

    # the preset is installed under the discovered key and loads back
    preset_file = paths.presets_dir() / f"{GAME}.json"
    assert Path(rec["preset"]) == preset_file
    assert load_preset(GAME, project) is not None
