"""Cutpoint never modifies, moves or deletes source footage, and writes nothing next to it.

End to end where it is cheap: a real ``analyze`` + ``clips`` run on a generated video, the stage
folder, and the 360 stitch with a fake SDK.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import needs_ffmpeg, set_config, write
from cutpoint import cli
from cutpoint.pano import stitch
from cutpoint.pano.project import sdk_output
from cutpoint.stage import STAGE_INDEX, clear_stage, stage_sources, staged_path
from pano_helpers import INSV, LRV, x3clip


def _snapshot(folder: Path) -> dict[str, tuple[str, int, int]]:
    """Every file below ``folder``: relative name -> (sha1, size, mtime_ns)."""
    return {
        p.relative_to(folder).as_posix(): (
            hashlib.sha1(p.read_bytes()).hexdigest(),
            p.stat().st_size,
            p.stat().st_mtime_ns,
        )
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# ------------------------------------------------------------------------ analyze + clips
@pytest.mark.slow
@needs_ffmpeg
def test_analyze_and_clips_leave_the_source_and_its_folder_untouched(video, home):
    before = _snapshot(video.parent)
    assert list(before) == [video.name]

    # The event steps (reactions, keyframes, crashes) take minutes with the full extras; they
    # only read frames and audio through the same decoders analyze uses.
    argv = ["analyze", str(video), "--no-vision", "--no-scenes", "--out", "safe"]
    argv += ["--skip", "reactions", "keyframes", "crashes"]
    assert cli.main(argv) == 0
    assert cli.main(["clips", "safe", "--top", "1"]) == 0

    clips_dir = home / "projects" / "safe" / "clips"
    assert any(clips_dir.glob("001_*.mp4"))  # the run did export something
    assert _snapshot(video.parent) == before  # bytes, mtime and folder listing unchanged


# ----------------------------------------------------------------------------------- stage
def _footage(folder: Path) -> list[Path]:
    return [write(folder / "a.mp4", b"a" * 1000), write(folder / "sub" / "b.mp4", b"b" * 2000)]


@pytest.mark.parametrize("targeted", [True, False], ids=["targeted", "everything"])
def test_stage_and_clear_stage_never_touch_the_originals(home, tmp_path, targeted):
    footage = tmp_path / "footage"
    originals = _footage(footage)
    before = _snapshot(footage)

    assert stage_sources([footage], recursive=True, min_free_gb=0)["files"] == 2
    copies = [Path(staged_path(str(p))) for p in originals]
    assert all(c.is_file() and c not in originals for c in copies)
    assert _snapshot(footage) == before

    removed = clear_stage([footage], recursive=True) if targeted else clear_stage()

    assert removed == 2
    assert not any(c.exists() for c in copies)
    assert _snapshot(footage) == before


def test_clear_stage_with_the_stage_folder_set_to_the_footage_folder(home, tmp_path):
    """A user who points ``stage_dir`` at the footage itself loses only the copies."""
    footage = tmp_path / "footage"
    originals = _footage(footage)
    before = _snapshot(footage)
    set_config(home, stage_dir=str(footage))

    assert stage_sources([footage], recursive=True, min_free_gb=0)["files"] == 2
    copies = [Path(staged_path(str(p))) for p in originals]
    assert all(c.parent == footage and c not in originals for c in copies)

    assert clear_stage() == 2

    after = _snapshot(footage)
    assert {k: v for k, v in after.items() if k in before} == before
    assert set(after) - set(before) == {STAGE_INDEX}  # only the (now empty) index remains


# ----------------------------------------------------------------------------- 360 stitch
FAKE_SDK = r"""
import sys
mode, args = sys.argv[1], sys.argv[2:]
part = args[args.index("-output") + 1]
for i in range(TAIL + 10):  # more lines than the ERROR tail, so the log must hold the rest
    print(f"synthetic sdk line {i:02d}")
print(f"{PROGRESS} = 100%", flush=True)
with open(part, "wb") as f:
    f.write(b"\0" * 1_200_000)
if mode == "error":
    print("error: synthetic failure one")
    print("error: synthetic failure two", flush=True)
"""


@pytest.fixture
def fake_sdk(tmp_path, monkeypatch):
    """``stitch_clips`` runs a fake SDK; returns ``set(mode)`` (``ok`` or ``error``)."""
    script = tmp_path / "fake_sdk.py"
    script.write_text(
        f"TAIL = {stitch.TAIL_LINES}\nPROGRESS = {stitch.PROGRESS_KEY!r}\n{FAKE_SDK}",
        encoding="utf-8",
    )
    exe = write(tmp_path / "sdk" / "MediaSDKTest.exe", b"")  # never run itself
    real_popen = subprocess.Popen
    mode = {"value": "ok"}

    def popen(cmd, *args, **kwargs):
        assert cmd[0] == str(exe)
        return real_popen([sys.executable, str(script), mode["value"], *cmd[1:]], *args, **kwargs)

    monkeypatch.setattr(stitch, "require_sdk", lambda: exe)
    monkeypatch.setattr(stitch, "MIN_FREE_GB", 0)
    monkeypatch.setattr(stitch, "_verify", lambda clip, path: "ok")  # no ffprobe on fake bytes
    monkeypatch.setattr(stitch.subprocess, "Popen", popen)
    return lambda value: mode.update(value=value)


def _camera(folder: Path):
    """An X3 clip (two ``.insv`` and an ``.lrv``) recorded into ``folder``."""
    for i, name in enumerate((*INSV, LRV)):
        write(folder / name, bytes([i]) * 4096)
    return x3clip(
        insv=[str(folder / n) for n in INSV], lrv=str(folder / LRV), frames=30, duration=1.0
    )


@pytest.mark.slow
@pytest.mark.parametrize("next_to_footage", [False, True], ids=["default-dir", "camera-folder"])
def test_stitch_never_touches_the_camera_files(home, tmp_path, fake_sdk, next_to_footage):
    camera = tmp_path / "camera"
    clip = _camera(camera)
    before = _snapshot(camera)
    if next_to_footage:
        set_config(home, pano={"stitched_dir": str(camera)})

    fake_sdk("error")
    [failed] = stitch.stitch_clips([clip])
    fake_sdk("ok")
    [ok] = stitch.stitch_clips([clip], force=True)

    assert failed["status"] == "failed" and ok["status"] == "ok"
    out = sdk_output(clip)
    assert out.is_file() and ok["output"] == str(out)
    after = _snapshot(camera)
    assert {k: v for k, v in after.items() if k in before} == before  # .insv/.lrv identical
    added = set(after) - set(before)
    if next_to_footage:
        stem = out.name.removesuffix(".equirect.mp4")
        assert added == {out.name, out.with_suffix(".json").name, f"logs/{stem}.log"}
    else:
        assert added == set()
        assert out.parent == home / "360-stitched"


@pytest.mark.slow
def test_failed_stitch_keeps_the_sdk_output_in_the_log_file(home, tmp_path, fake_sdk, caplog):
    clip = _camera(tmp_path / "camera")
    fake_sdk("error")

    with caplog.at_level("ERROR"):
        [rec] = stitch.stitch_clips([clip])

    log_file = Path(rec["log_file"])
    text = log_file.read_text(encoding="utf-8")
    for i in range(stitch.TAIL_LINES + 10):  # every SDK line, not just the tail
        assert f"synthetic sdk line {i:02d}" in text
    [error] = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert "error: synthetic failure one" in error
    assert "error: synthetic failure two" in error
    assert str(log_file) in error
