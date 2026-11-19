"""Robustness fixes of audit #1085, A250-A260."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from cutpoint import config, ffmpeg_utils, imageio, stage, tsrepair
from cutpoint.errors import FFmpegError, FFmpegNotFoundError, FolderBusyError


# -- A250: unique temp names
def test_unique_tmp_differs_and_is_found(tmp_path: Path) -> None:
    dst = tmp_path / "audio16k.wav"
    a, b = ffmpeg_utils.unique_tmp(dst), ffmpeg_utils.unique_tmp(dst)
    assert a != b
    assert a.parent == tmp_path
    assert a.name.endswith(".tmp.wav")
    a.write_bytes(b"x")
    assert ffmpeg_utils.tmp_files(dst) == [a]


def test_extract_audio_uses_a_unique_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(args: list[str], *_a: object, **_k: object) -> str:
        Path(args[-1]).write_bytes(b"wav")
        seen.append(args[-1])
        return "ok"

    monkeypatch.setattr(ffmpeg_utils, "run_progress", fake)
    monkeypatch.setattr(ffmpeg_utils, "_tracks_or_every", lambda *_a: [0])
    info = ffmpeg_utils.MediaInfo.__new__(ffmpeg_utils.MediaInfo)
    object.__setattr__(info, "audio_streams", 1)
    object.__setattr__(info, "duration", 10.0)
    dst = tmp_path / "audio16k.wav"
    ffmpeg_utils.extract_audio(tmp_path / "a.mp4", dst, info)
    ffmpeg_utils.extract_audio(tmp_path / "a.mp4", dst, info)
    assert seen[0] != seen[1]
    assert dst.read_bytes() == b"wav"
    assert ffmpeg_utils.tmp_files(dst) == []


# -- A251: probes go through _exec with a timeout
def test_read_probes_use_exec_with_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float | None] = []

    def fake(tool: str, args: object, timeout: float | None, path: object) -> object:
        calls.append(timeout)
        return subprocess.CompletedProcess([], 0, b"", b"[Parsed_cropdetect] crop=100:50:2:4")

    monkeypatch.setattr(ffmpeg_utils, "_exec", fake)
    assert ffmpeg_utils.read_window_ok(Path("a.mp4"), 1.0) is True
    assert ffmpeg_utils.cropdetect_at(Path("a.mp4"), 1.0) == (100, 50, 2, 4)
    assert calls == [ffmpeg_utils.READ_PROBE_TIMEOUT_S] * 2


def test_read_probes_treat_a_timeout_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*_a: object) -> object:
        raise FFmpegError("ffmpeg timed out after 120s on a.mp4")

    monkeypatch.setattr(ffmpeg_utils, "_exec", hang)
    assert ffmpeg_utils.read_window_ok(Path("a.mp4"), 1.0) is False
    assert ffmpeg_utils.cropdetect_at(Path("a.mp4"), 1.0) is None


# -- A252: _probe
def _probe_with(monkeypatch: pytest.MonkeyPatch, streams: list[dict]) -> ffmpeg_utils.MediaInfo:
    data = {"streams": streams, "format": {"duration": "10"}}
    monkeypatch.setattr(ffmpeg_utils, "ffprobe_json", lambda *_a, **_k: data)
    return ffmpeg_utils._probe(Path("a.mp4"))


def _video(**kw: object) -> dict:
    return {"codec_type": "video", "width": 640, "height": 360, **kw}


def test_probe_zero_rates_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    info = _probe_with(monkeypatch, [_video(avg_frame_rate="0/0", r_frame_rate="0/0")])
    assert float(info.fps) == 30.0


def test_probe_skips_cover_art(monkeypatch: pytest.MonkeyPatch) -> None:
    cover = _video(width=500, height=500, disposition={"attached_pic": 1}, avg_frame_rate="0/0")
    info = _probe_with(monkeypatch, [cover, _video(avg_frame_rate="25/1")])
    assert (info.width, info.height) == (640, 360)
    assert float(info.fps) == 25.0
    only = _probe_with(monkeypatch, [cover])  # nothing else: the picture is still read
    assert only.width == 500


# -- A253: read_image
def test_read_image_empty_and_garbage(tmp_path: Path) -> None:
    empty = tmp_path / "e.jpg"
    empty.write_bytes(b"")
    junk = tmp_path / "j.jpg"
    junk.write_bytes(b"not a picture")
    assert imageio.read_image(empty) is None
    assert imageio.read_image(junk) is None
    good = tmp_path / "g.png"
    assert imageio.write_png(good, np.zeros((4, 4, 3), np.uint8))
    assert imageio.read_image(good) is not None


# -- A254: a pinned tool that is not a file
def test_find_tool_env_naming_no_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUTPOINT_FFPROBE", str(tmp_path / "nope"))
    ffmpeg_utils.find_tool.cache_clear()
    try:
        with pytest.raises(FFmpegNotFoundError, match="CUTPOINT_FFPROBE"):
            ffmpeg_utils.find_tool("ffprobe")
        real = tmp_path / "ffprobe"
        real.write_bytes(b"")
        monkeypatch.setenv("CUTPOINT_FFPROBE", str(real))
        ffmpeg_utils.find_tool.cache_clear()
        assert ffmpeg_utils.find_tool("ffprobe") == str(real)
    finally:
        ffmpeg_utils.find_tool.cache_clear()


# -- A255: concurrent config changes both survive
def test_update_config_concurrent(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    keys = [f"k{i}" for i in range(12)]

    def put(key: str) -> None:
        config.update_config(lambda _data: ({key: True}, ()), path)

    threads = [threading.Thread(target=put, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(json.loads(path.read_text(encoding="utf-8"))) == set(keys)


# -- A256: the copy lock wait has a deadline
def test_copy_lock_gives_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(*_a: object, **_k: object) -> None:
        raise FolderBusyError("another run holds the lock")

    monkeypatch.setattr(tsrepair.folders, "swap_lock", busy)
    monkeypatch.setattr(tsrepair, "COPY_WAIT_S", 0.3)
    with pytest.raises(FolderBusyError), tsrepair._copy_lock(tmp_path, tmp_path / "a.ts"):
        pytest.fail("should not get the lock")


# -- A257: no .partial after a failed copy
def test_copy_once_leaves_no_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x" * 100)
    dst = tmp_path / "out" / "a.mp4"
    dst.parent.mkdir()

    def boom(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(stage.os, "utime", boom)
    with pytest.raises(KeyboardInterrupt):
        stage.copy_once(src, dst, "a")
    assert list(dst.parent.iterdir()) == []


# -- A258: a frame without a showinfo line
def test_pts_time_missing_line_continues_from_previous(caplog: pytest.LogCaptureFixture) -> None:
    log = ffmpeg_utils._StderrLog(True, None)
    log.times = [10.0, 10.5, float("nan")]  # frame 2 has no line; the end marker
    assert log.pts_time(0) == 10.0
    assert log.pts_time(1) == 10.5
    with caplog.at_level(logging.WARNING):
        assert log.pts_time(2) == pytest.approx(11.0)
    assert "No presentation time" in caplog.text


# -- A259: Config keywords
def test_config_booleans_land_on_their_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"vision": True, "visual_index": False, "ocr_det_skip": True, "vision_detect_all": True}
        ),
        encoding="utf-8",
    )
    cfg = config._load_config(path)
    assert (cfg.vision, cfg.visual_index, cfg.ocr_det_skip, cfg.vision_detect_all) == (
        True,
        False,
        True,
        True,
    )


# -- A260: child_scope restores the outer scope
def test_child_scope_nests() -> None:
    outer, inner = ffmpeg_utils.ChildScope(), ffmpeg_utils.ChildScope()
    with ffmpeg_utils.child_scope(outer):
        with ffmpeg_utils.child_scope(inner):
            assert ffmpeg_utils._SCOPE.current is inner
        assert ffmpeg_utils._SCOPE.current is outer
    assert ffmpeg_utils._SCOPE.current is None


# -- F2: temp names carry the process id; temps of gone processes are swept
def _host() -> str:
    return ffmpeg_utils._host_tag()


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_tmp_name_has_this_pid_and_sweep_keeps_live_removes_dead(tmp_path: Path) -> None:
    dst = tmp_path / "audio16k.wav"
    mine = ffmpeg_utils.unique_tmp(dst)
    assert f".{os.getpid()}." in mine.name
    dead = tmp_path / f"audio16k.{_dead_pid()}.{_host()}.deadbeef.tmp.wav"
    for p in (mine, dead):
        p.write_bytes(b"x")
    ffmpeg_utils.sweep_tmp(dst)
    assert mine.exists()
    assert not dead.exists()


def test_extract_audio_sweeps_dead_temps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dead = tmp_path / f"audio16k.{_dead_pid()}.{_host()}.deadbeef.tmp.wav"
    dead.write_bytes(b"x")
    monkeypatch.setattr(
        ffmpeg_utils, "run_progress", lambda args, *_a, **_k: Path(args[-1]).write_bytes(b"w")
    )
    monkeypatch.setattr(ffmpeg_utils, "_tracks_or_every", lambda *_a: [0])
    info = ffmpeg_utils.MediaInfo.__new__(ffmpeg_utils.MediaInfo)
    object.__setattr__(info, "audio_streams", 1)
    object.__setattr__(info, "duration", 10.0)
    ffmpeg_utils.extract_audio(tmp_path / "a.mp4", tmp_path / "audio16k.wav", info)
    assert not dead.exists()


def test_tmp_files_own_ignores_a_foreign_temp(tmp_path: Path) -> None:
    dst = tmp_path / "audio16k.wav"
    mine = ffmpeg_utils.unique_tmp(dst)
    foreign = tmp_path / f"audio16k.1.{_host()}.cafecafe.tmp.wav"
    for p in (mine, foreign):
        p.write_bytes(b"x")
    assert ffmpeg_utils.tmp_files(dst, own=True) == [mine]
    assert set(ffmpeg_utils.tmp_files(dst)) == {mine, foreign}


def test_watch_extract_ignores_a_foreign_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint import analyze  # noqa: PLC0415

    monkeypatch.setattr(analyze, "EXTRACT_POLL_S", 0.01)
    wav = tmp_path / "audio16k.wav"
    (tmp_path / f"audio16k.1.{_host()}.cafecafe.tmp.wav").write_bytes(
        bytes(10_000_000)
    )  # another run's
    got: list[float] = []
    with analyze._watch_extract(wav, 10.0, got.append):
        time.sleep(0.2)
    assert got == []


# -- F4: a missing ffmpeg is not swallowed by the probes
def test_read_probes_let_a_missing_ffmpeg_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_a: object) -> object:
        raise FFmpegNotFoundError("ffmpeg not found")

    monkeypatch.setattr(ffmpeg_utils, "_exec", missing)
    with pytest.raises(FFmpegNotFoundError):
        ffmpeg_utils.read_window_ok(Path("a.mp4"), 1.0)
    with pytest.raises(FFmpegNotFoundError):
        ffmpeg_utils.cropdetect_at(Path("a.mp4"), 1.0)


# -- F5: the 30 fps assumption is logged
def test_probe_zero_rates_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _probe_with(monkeypatch, [_video(avg_frame_rate="0/0", r_frame_rate="0/0")])
    assert "a.mp4" in caplog.text
    assert "30 fps" in caplog.text


# -- F8: a first frame without a time line is at 0
def test_pts_time_first_frame_missing_is_zero() -> None:
    log = ffmpeg_utils._StderrLog(True, None)
    log.times = [float("nan")]
    assert log.pts_time(0) == 0.0


# -- F7: pictures are written whole
def test_write_png_replaces_atomically_and_leaves_no_temp(tmp_path: Path) -> None:
    out = tmp_path / "p.png"
    out.write_bytes(b"old")
    assert imageio.write_png(out, np.zeros((4, 4, 3), np.uint8))
    assert imageio.read_image(out) is not None
    assert [p.name for p in tmp_path.iterdir()] == ["p.png"]


def test_sweep_keeps_a_temp_of_another_computer(tmp_path: Path) -> None:
    dst = tmp_path / "audio16k.wav"
    foreign = tmp_path / f"audio16k.{_dead_pid()}.00000000.deadbeef.tmp.wav"
    foreign.write_bytes(b"x")
    ffmpeg_utils.sweep_tmp(dst)
    assert foreign.exists()  # its process cannot be told from here


def test_sweep_removes_the_fixed_temp_of_older_versions(tmp_path: Path) -> None:
    dst = tmp_path / "audio16k.wav"
    old = tmp_path / "audio16k.tmp.wav"
    done = tmp_path / "audio16k.wav"
    old.write_bytes(b"x")
    done.write_bytes(b"keep")
    ffmpeg_utils.sweep_tmp(dst)
    assert not old.exists()
    assert done.read_bytes() == b"keep"
    proxy_old = tmp_path / "proxy.tmp.mp4"
    proxy_old.write_bytes(b"x")
    ffmpeg_utils.sweep_tmp(tmp_path / "proxy.mp4")
    assert not proxy_old.exists()


def test_write_png_retries_a_locked_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.replace
    calls: list[int] = []

    def flaky(src: object, dst: object) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(13, "in use")
        real(src, dst)

    monkeypatch.setattr(imageio.cache.os, "replace", flaky)
    out = tmp_path / "p.png"
    assert imageio.write_png(out, np.zeros((4, 4, 3), np.uint8))
    assert len(calls) == 2
    assert imageio.read_image(out) is not None
