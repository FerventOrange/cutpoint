"""Helpers shared by the scene-pass test modules (test_scene_*.py)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, write
from cutpoint import analyze, ffmpeg_utils

CUDA = ffmpeg_utils.HwDecoder("cuda")


def nvdec_source(monkeypatch: pytest.MonkeyPatch, duration: float) -> None:
    """Every source probes as ``duration`` seconds of H.264 and NVDEC decodes it."""
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, duration))
    monkeypatch.setattr(analyze, "decoder_for", lambda src: CUDA)


def fake_source_io(
    monkeypatch: pytest.MonkeyPatch, duration: float = 60.0, *, flat_lufs: float | None = None
) -> None:
    """What ``analyze_source`` reads around the scene pass, faked: ffprobe (``duration`` s, one
    audio stream), the audio extraction, the read check and the loudness curve (every 0.5 s:
    one loud burst at 20 s, or ``flat_lufs`` throughout)."""
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, duration, audio_streams=1))
    monkeypatch.setattr(analyze, "extract_audio", lambda io, wav, *a, **k: write(wav, "wav"))
    monkeypatch.setattr(analyze, "check_read", lambda *a, **k: None)
    times = np.arange(0.0, duration, 0.5)
    if flat_lufs is None:
        lufs = -50.0 + 35.0 * np.exp(-(((times - 20.0) / 3.0) ** 2))
    else:
        lufs = np.full(len(times), flat_lufs)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (times, lufs))


def near(got: list[float], want: list[float]) -> bool:
    """``got`` and ``want`` hold the same cuts within one 0.1 s proxy frame."""
    return len(got) == len(want) and all(
        abs(a - b) <= analyze.SCENE_GRID_S + 1e-6 for a, b in zip(got, want, strict=True)
    )


def concat_clip(
    out: Path, parts: list[tuple[str, float]], *codec: str, fmt: str | None = None
) -> Path:
    """A 160x120 10 fps clip of the lavfi sources ``parts`` (name, seconds) back to back, so it
    has a hard cut at each join; encoded with ``codec`` (and muxed as ``fmt``) by real ffmpeg."""
    args: list[str] = []
    for name, dur in parts:
        sep = ":" if "=" in name else "="
        args += ["-f", "lavfi", "-i", f"{name}{sep}size=160x120:rate=10:duration={dur}"]
    graph = "".join(f"[{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1[v]"
    fmt_args = ["-f", fmt] if fmt else []
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", *args, "-filter_complex", graph, "-map", "[v]",
         *codec, *fmt_args, str(out)],
        check=True,
    )  # fmt: skip
    return out


def late_copy(src: Path, out: Path, acodec: str) -> Path:
    """``src`` (24 s) plus a sine audio track, its video 0.25 s late."""
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-itsoffset", "0.25", "-i", str(src), "-f",
         "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=24", "-map", "0:v",
         "-map", "1:a", "-c:v", "copy", "-c:a", acodec, str(out)],
        check=True,
    )  # fmt: skip
    return out
