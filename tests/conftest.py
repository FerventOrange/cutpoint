"""Isolate every test from the real user data (CUTPOINT_HOME, env, LOCALAPPDATA)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

from cutpoint import analyze, clips, config, paths, reset_state
from cutpoint.ffmpeg_utils import MediaInfo


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point CUTPOINT_HOME and every platform default at ``tmp_path``; yields the home."""
    for name in list(os.environ):
        if name.startswith("CUTPOINT_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    monkeypatch.setenv("CUTPOINT_HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setattr(config, "_drive_remote", lambda d: None)  # no real mapped drives
    reset_state()
    assert paths.home() == home, "isolation failed: CUTPOINT_HOME not honoured"
    yield home
    reset_state()


@pytest.fixture
def home(isolated_env: Path) -> Path:
    return isolated_env


def write(path: Path, text: str | bytes = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def set_config(home: Path, **data) -> None:
    """Write ``config.json`` under ``home`` and drop the cached copy."""
    write(home / "config.json", json.dumps(data))
    config.get_config.cache_clear()


def remap(home: Path, monkeypatch: pytest.MonkeyPatch, old, new, exclude=()) -> None:
    """``source_remap`` from ``old`` to ``new``; paths compare as plain spellings, on any OS."""
    data = {"source_remap": {str(old): str(new)}}
    if exclude:
        data["source_remap_exclude"] = list(exclude)
    set_config(home, **data)
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))


def media_info(path: Path = Path("v.mp4"), duration: float = 100.0, **over) -> MediaInfo:
    """A 1920x1080, 30 fps, H.264 + AAC 48 kHz stereo probe result; ``over`` replaces fields."""
    fields = {
        "path": path, "duration": duration, "vduration": duration, "width": 1920, "height": 1080,
        "fps": Fraction(30), "pix_fmt": "yuv420p", "vcodec": "h264", "audio_streams": 1,
        "acodec": "aac", "sample_rate": 48000, "channels": 2,
    }  # fmt: skip
    return MediaInfo(**(fields | over))


_skip_without_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
)


def needs_ffmpeg(test):
    """Skip without ffmpeg/ffprobe, and mark the test ``ffmpeg`` (it runs the real tools)."""
    return pytest.mark.ffmpeg(_skip_without_ffmpeg(test))


KEYS = {"ok", "command", "schema", "project", "outputs", "warnings", "error"}


def envelope(out: str) -> dict:
    """``out`` must be exactly one JSON object (and a newline)."""
    doc, end = json.JSONDecoder().raw_decode(out)
    assert out[end:] == "\n"
    assert isinstance(doc, dict) and set(doc) == KEYS
    assert doc["schema"] == 1 and isinstance(doc["ok"], bool)
    assert isinstance(doc["command"], str) and isinstance(doc["outputs"], dict)
    assert isinstance(doc["warnings"], list)
    assert doc["project"] is None or isinstance(doc["project"], str)
    if doc["ok"]:
        assert doc["error"] is None
    else:
        assert set(doc["error"]) == {"code", "message"}
    return doc


def run_json(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run ``cutpoint <argv> --json``: exit code and the one envelope on stdout."""
    from cutpoint import cli  # noqa: PLC0415

    code = cli.main([*argv, "--json"])
    return code, envelope(capsys.readouterr().out)


def warning_codes(doc_or_warnings: dict | list[dict]) -> list[str]:
    """The ``code`` of each warning in an envelope (or in its ``warnings`` list)."""
    warnings = doc_or_warnings["warnings"] if isinstance(doc_or_warnings, dict) else doc_or_warnings
    return [w["code"] for w in warnings]


def link_file(link: Path, target: Path) -> None:
    """Make ``link`` a file symlink to ``target`` (skip where files cannot be linked)."""
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file symlink here")


def link_dir(link: Path, target: Path) -> None:
    """Make ``link`` a directory link to ``target``: a symlink, or a junction on Windows."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot create a directory symlink here")
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
    )
    if done.returncode != 0:
        pytest.skip("cannot create a symlink or junction here")


def hard_link(link: Path, target: Path) -> None:
    try:
        os.link(target, link)
    except (OSError, NotImplementedError, AttributeError) as err:
        pytest.skip(f"hard links are not supported here: {err}")


def messages(caplog: pytest.LogCaptureFixture, level: str = "WARNING", logger=None) -> list[str]:
    """Messages logged at exactly ``level`` (optionally only by logger name ``logger``)."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == level and (logger is None or r.name == logger)
    ]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Opt-in: ``time.sleep`` records its argument instead of waiting; returns the list."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch):
    """Returns ``set(flag)``, which makes ``clips._WINDOWS`` report Windows (or not)."""
    return lambda flag=True: monkeypatch.setattr(clips, "_WINDOWS", flag)


@pytest.fixture
def moved(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A 360 clip recorded under ``old/`` whose footage now lives in ``new/`` (``source_remap``)."""
    from pano_helpers import INSV, LRV, x3clip  # noqa: PLC0415

    old, new = tmp_path / "old", tmp_path / "new"
    for name in (*INSV, LRV):
        write(new / name)  # the old folder is gone; no stitched file anywhere
    remap(home, monkeypatch, old, new)
    return x3clip(lrv=str(old / LRV), insv=[str(old / n) for n in INSV]), old, new


VIDEO_SEGMENTS = [
    {
        "start": 3.0,
        "end": 4.5,
        "text": " ha ha that was great",
        "words": [
            {"start": 3.0, "end": 3.3, "word": " ha"},
            {"start": 3.3, "end": 3.6, "word": " ha"},
            {"start": 3.6, "end": 3.9, "word": " that"},
            {"start": 3.9, "end": 4.1, "word": " was"},
            {"start": 4.1, "end": 4.5, "word": " great"},
        ],
    }
]


class StubTranscriber:
    """No whisper: every source "says" VIDEO_SEGMENTS."""

    model_name, device = "stub", "cpu"

    def __init__(self, language: str | None = None, **_kw) -> None:
        self.language = language

    def transcribe(self, _wav: Path) -> list[dict]:
        return VIDEO_SEGMENTS

    def close(self) -> None:
        pass


@pytest.fixture
def video(home, tmp_path, monkeypatch) -> Path:
    """A 10 s 160x120 video, quiet with two loud bursts; the transcriber stubbed.

    Does not skip by itself: tests using it need ``@needs_ffmpeg``.
    """
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    out = tmp_path / "videos" / "clip.mp4"
    out.parent.mkdir()
    loud = "volume=enable='between(t,2,4.5)+between(t,7,8.5)':volume=30"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=10:duration=10", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=10", "-af", f"volume=0.02,{loud}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )  # fmt: skip
    return out
