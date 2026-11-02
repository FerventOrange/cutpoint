"""Real clips for the assemble tests: three spans of one synthetic BT.709 source, exported by
``encode.export_clip`` with a manifest. Needs ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from cutpoint import encode
from cutpoint.ffmpeg_utils import probe

SPANS = [(0.5, 5.5), (6.0, 10.2), (10.5, 15.5)]  # source seconds of the three clips
SHORT = [(0.5, 4.5), (5.0, 8.5), (9.0, 12.0)]  # 4, 3.5 and 3 s


def export_clips(tmp_path: Path, rate: str = "30", spans=SPANS) -> Path:
    src = tmp_path / "src.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=320x180:rate={rate}:duration=16,"
         "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
         "-f", "lavfi", "-i", "sine=frequency=330:beep_factor=3:sample_rate=48000:duration=16",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc",
         "bt709", "-colorspace", "bt709", "-color_range", "tv", "-c:a", "aac", str(src)],
        check=True,
    )  # fmt: skip
    clips = tmp_path / "clips"
    clips.mkdir()
    fps = round(float(Fraction(rate)))
    info, fmt, entries = probe(src), encode.OutputFormat(320, 180, fps), []
    if rate != str(fps):  # export_clip writes the rate into the filter: keep 30000/1001
        fmt = encode.OutputFormat(320, 180, rate)  # type: ignore[arg-type]
    for n, (a, b) in enumerate(spans, 1):
        name = f"{n:03d}_m.mp4"
        encode.export_clip(src, info, a, b, clips / name, fmt, False, audio_fade=0.25)
        entries.append({"file": name, "source": str(src), "start": a, "end": b, "audio_fade": 0.25})
    (clips / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    return clips
