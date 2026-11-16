"""Short MPEG-TS parts and their binary joins, the media of the tests on timestamp jumps."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PART_S = 4  # length of one part (s)


def make_part(
    d: Path,
    name: str,
    *,
    off: float = 0.0,
    audio: bool = True,
    fmt: str = "mpegts",
    tracks: int = 1,
) -> Path:
    """One short part; ``off`` shifts its timestamps (``-output_ts_offset``); ``tracks``
    audio tracks, each a sine of its own pitch."""
    out = d / name
    cmd = [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc=size=128x96:rate=10:duration={PART_S}"]  # fmt: skip
    for i in range(tracks if audio else 0):
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency={440 * (i + 1)}:sample_rate=16000:duration={PART_S}"]  # fmt: skip
    cmd += ["-c:v", "mpeg2video" if fmt == "mpeg" else "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        if tracks > 1:
            cmd += ["-map", "0:v", *[a for j in range(tracks) for a in ("-map", f"{j + 1}:a")]]
        cmd += ["-c:a", "mp2" if fmt == "mpeg" else "aac", "-b:a", "32k"]
    cmd += ["-output_ts_offset", str(off), "-f", fmt, str(out)]
    subprocess.run(cmd, check=True)
    return out


def join_parts(out: Path, *parts: Path) -> Path:
    out.write_bytes(b"".join(p.read_bytes() for p in parts))
    return out
