"""ffmpeg_utils: tools are never found in the current folder (A664); inputs open only local
files and pipes (A665)."""

from __future__ import annotations

import os
import stat

import pytest

from conftest import write
from cutpoint import ffmpeg_utils as fu

WL = ["-protocol_whitelist", fu.INPUT_PROTOCOLS]


def test_every_input_but_lavfi_is_limited_to_files_and_pipes() -> None:
    cmd = ["ffmpeg", "-y", "-ss", "1", "-i", "a.mp4", "-f", "lavfi", "-i", "anullsrc"]
    cmd += ["-f", "concat", "-safe", "0", "-i", "l.txt", "-f", "mp4", "out.mp4"]
    assert fu.limit_protocols(cmd) == [
        "ffmpeg", "-y", "-ss", "1", *WL, "-i", "a.mp4", "-f", "lavfi", "-i", "anullsrc",
        "-f", "concat", "-safe", "0", *WL, "-i", "l.txt", "-f", "mp4", "out.mp4",
    ]  # fmt: skip
    own = ["ffmpeg", "-protocol_whitelist", "file", "-i", "a.mp4", "o.mp4"]
    assert fu.limit_protocols(own) == own


def test_command_limits_the_protocols(monkeypatch) -> None:
    monkeypatch.setattr(fu, "find_tool", lambda name: name)
    assert fu._command("ffmpeg", ["-i", "a.mp4", "o.mp4"]) == [
        "ffmpeg",
        *WL,
        "-i",
        "a.mp4",
        "o.mp4",
    ]


def _exe(path):
    """An executable ``path`` (``.exe`` added on Windows, where ``shutil.which`` needs it)."""
    if os.name == "nt":
        path = path.with_name(path.name + ".exe")
    write(path, "#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def fresh_find_tool(monkeypatch):
    monkeypatch.delenv("CUTPOINT_FFMPEG", raising=False)
    fu.find_tool.cache_clear()
    yield
    fu.find_tool.cache_clear()


def test_a_tool_in_a_relative_path_folder_is_not_used(tmp_path, monkeypatch, fresh_find_tool):
    monkeypatch.chdir(tmp_path)
    _exe(tmp_path / "ffmpeg")  # planted in the current folder, which PATH names as "."
    real = _exe(tmp_path / "bin" / "ffmpeg")
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", str(real.parent)]))
    assert os.path.normcase(fu._on_path("ffmpeg") or "") == os.path.normcase(str(real))
    monkeypatch.setenv("PATH", ".")
    assert fu._on_path("ffmpeg") is None


def test_no_localappdata_searches_no_winget_folder_below_the_current_one(
    tmp_path, monkeypatch, fresh_find_tool
):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", "")
    monkeypatch.setattr(fu.sys, "platform", "win32")
    with pytest.raises(fu.FFmpegNotFoundError):
        fu.find_tool("ffmpeg")


def test_the_peak_tag_probe_names_its_input_with_i(monkeypatch, tmp_path) -> None:
    """Review F4: ffprobe's input is given with -i, so it gets the protocol whitelist."""
    from cutpoint import assemble  # noqa: PLC0415

    seen: list[list[str]] = []
    monkeypatch.setattr(fu, "find_tool", lambda name: name)
    monkeypatch.setattr(
        assemble, "run", lambda args: seen.append(fu._command(args[0], args[1:])) or ""
    )
    clip = tmp_path / "c.mp4"
    assemble._peak_tagged(clip)
    (cmd,) = seen
    assert cmd[-4:] == [*WL, "-i", str(clip)]
