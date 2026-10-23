"""Exports appear under their final name only when complete."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from cutpoint import assemble, cli, clips, encode, extend, ffmpeg_utils, planning
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo


def _info(path: Path) -> MediaInfo:
    return MediaInfo(
        path=path, duration=100.0, vduration=100.0, width=1920, height=1080,
        fps=Fraction(30), pix_fmt="yuv420p", vcodec="h264", audio_streams=1,
        acodec="aac", sample_rate=48000, channels=2,
    )  # fmt: skip


def _dies_midway(exc: BaseException):
    def run(args, **_kw):
        Path(args[-1]).write_bytes(b"truncated")  # ffmpeg had started writing
        raise exc

    return run


FAILURES = [FFmpegError("ffmpeg failed (exit 1)"), KeyboardInterrupt()]


@pytest.mark.parametrize("exc", FAILURES, ids=["ffmpeg-error", "ctrl-c"])
def test_failed_or_interrupted_clip_export_leaves_no_file(monkeypatch, tmp_path, exc):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(encode, "run", _dies_midway(exc))
    src = tmp_path / "src.mp4"
    with pytest.raises(type(exc)):
        encode.export_clip(
            src,
            _info(src),
            0.0,
            2.0,
            tmp_path / "001.mp4",
            encode.OutputFormat(1920, 1080, 30),
            False,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("exc", FAILURES, ids=["ffmpeg-error", "ctrl-c"])
def test_failed_or_interrupted_assemble_leaves_no_final_video(monkeypatch, tmp_path, exc):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "001.mp4").write_bytes(b"x")
    monkeypatch.setattr(assemble, "probe", _info)
    monkeypatch.setattr(assemble, "baked_fades", lambda d, files: [0.0] * len(files))

    def concat_copy(files, out, infos, fades):
        out.write_bytes(b"truncated")
        raise exc

    monkeypatch.setattr(assemble, "concat_copy", concat_copy)
    with pytest.raises(type(exc)):
        assemble.assemble(clips, tmp_path / "final.mp4")
    assert list(tmp_path.glob("*.mp4")) == [] and sorted(p.name for p in clips.iterdir()) == [
        "001.mp4"
    ]


@pytest.mark.skipif(os.name == "nt", reason="the UI kills Windows jobs with taskkill, not SIGTERM")
def test_sigterm_cancels_a_command_like_ctrl_c(monkeypatch):
    """A UI cancel (SIGTERM) unwinds as exit 130 and leaves the caller's handler alone."""

    def cancelled(_args):
        signal.raise_signal(signal.SIGTERM)
        return 0

    monkeypatch.setattr(cli, "_cmd_paths", cancelled)
    before = signal.getsignal(signal.SIGTERM)

    assert cli.main(["paths"]) == 130
    assert signal.getsignal(signal.SIGTERM) is before


# --- exit codes, failed-run cleanup, locked files ---


@pytest.mark.parametrize(
    ("rc", "shown"),
    [
        (3752568763, "-542398533 (0xDFABA7BB)"),  # Windows crash code, reported unsigned
        (-9, "-9"),  # POSIX: killed by SIGKILL
        (1, "1"),
    ],
)
def test_ffmpeg_failure_shows_crash_codes_readably(monkeypatch, rc, shown):
    monkeypatch.setattr(ffmpeg_utils, "find_tool", lambda name: name)
    monkeypatch.setattr(ffmpeg_utils, "stage_inputs", lambda cmd: cmd)
    monkeypatch.setattr(
        ffmpeg_utils.subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, rc, "", "boom"),
    )
    with pytest.raises(FFmpegError, match=rf"\(exit {re.escape(shown)}\)"):
        ffmpeg_utils.run(["ffmpeg", "-i", "x"])


def _project(home: Path) -> tuple[Path, Path]:
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    src = project / "src.mp4"
    src.write_bytes(b"x")
    seg = project / "seg.json"
    seg.write_text(
        json.dumps([{"source": str(src), "start": 0.0, "end": 2.0, "label": "a"}]),
        encoding="utf-8",
    )
    return project, seg


@pytest.mark.parametrize("preexisting", [False, True], ids=["created", "preexisting"])
def test_failed_export_leaves_no_new_empty_clips_folder(monkeypatch, home, preexisting):
    project, seg = _project(home)
    out = project / "clips"
    if preexisting:
        out.mkdir()

    def broken_probe(_path):
        raise FFmpegError("ffprobe failed")

    monkeypatch.setattr(clips, "probe", broken_probe)
    with pytest.raises(FFmpegError):
        clips.export_clips(project, segments=seg)
    assert out.is_dir() is preexisting


def _locked(winerror: int) -> PermissionError:
    err = PermissionError(13, "The process cannot access the file")
    err.winerror = winerror
    return err


def _fake_export(monkeypatch):
    """Export without ffmpeg: every clip is the bytes b"new"."""

    def export(src, info, start, end, out, *_a, **_kw):
        out.write_bytes(b"new")

    monkeypatch.setattr(clips, "probe", _info)
    monkeypatch.setattr(clips, "export_clip_checked", export)


@pytest.mark.parametrize("winerror", [32, 5])
def test_clips_force_over_a_locked_clip_says_who_to_close(monkeypatch, home, caplog, winerror):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    project, seg = _project(home)
    old = project / "clips" / "001_a.mp4"
    old.parent.mkdir()
    old.write_bytes(b"x")
    _fake_export(monkeypatch)
    real_replace = Path.replace

    def replace(self, target):
        if self == old:
            raise _locked(winerror)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    assert cli.main(["clips", "p", "--segments", str(seg), "--force"]) == 1
    assert "open in another program" in caplog.text
    assert "001_a.mp4" in caplog.text
    assert old.read_bytes() == b"x"


def test_clips_force_retries_a_lock_that_clears(monkeypatch, home):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    project, seg = _project(home)
    old = project / "clips" / "001_a.mp4"
    old.parent.mkdir()
    old.write_bytes(b"x")
    _fake_export(monkeypatch)
    real_replace = Path.replace
    attempts = []

    def replace(self, target):
        if self == old and not attempts:
            attempts.append(1)
            raise _locked(32)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    clips.export_clips(project, segments=seg, force=True)
    assert attempts and old.read_bytes() == b"new"


def test_locked_final_video_names_the_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    out = tmp_path / "final.mp4"

    def replace(self, target):
        raise _locked(32)

    monkeypatch.setattr(Path, "replace", replace)
    with (
        pytest.raises(PermissionError, match="open in another program") as exc,
        ffmpeg_utils.atomic_output(out) as part,
    ):
        part.write_bytes(b"x")
    assert str(out) in str(exc.value)
    assert not ffmpeg_utils.part_path(out).exists()


@pytest.mark.parametrize("winerror", [None, 2])
def test_non_lock_permission_error_is_reraised_unchanged(monkeypatch, tmp_path, winerror):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    out = tmp_path / "final.mp4"

    def replace(self, target):
        err = PermissionError(13, "denied")
        if winerror is not None:
            err.winerror = winerror
        raise err

    monkeypatch.setattr(Path, "replace", replace)
    with (
        pytest.raises(PermissionError) as exc,
        ffmpeg_utils.atomic_output(out) as part,
    ):
        part.write_bytes(b"x")
    assert "another program" not in str(exc.value)
    assert exc.value.errno == 13
    assert exc.value.strerror == "denied"
    assert not ffmpeg_utils.part_path(out).exists()


# --- all-or-nothing replace of a clips folder (clips --force, in-place extend --force)

NAMES = ["001_a.mp4", "002_b.mp4", "003_c.mp4"]


class _HardKill(BaseException):
    """Stands in for taskkill /F: nothing after it runs, the folder stays as it is."""


def _setup(home: Path, monkeypatch) -> Path:
    """Project with three old clips, a manifest, user notes and a stubbed encoder."""
    project = home / "projects" / "p"
    src = project / "src.mp4"
    clips_dir = project / "clips"
    clips_dir.mkdir(parents=True)
    src.write_bytes(b"x")
    (project / "seg.json").write_text(
        json.dumps(
            [{"source": str(src), "start": i * 5.0, "end": i * 5.0 + 2, "label": n[4]}
             for i, n in enumerate(NAMES)]
        ),
        encoding="utf-8",
    )  # fmt: skip
    entries = [{"file": n, "source": str(src), "start": 10.0, "end": 20.0} for n in NAMES]
    fmt = {"width": 1920, "height": 1080, "fps": 30, "vertical": False}
    for n in NAMES:
        (clips_dir / n).write_bytes(b"old " + n.encode())
    (clips_dir / "manifest.json").write_text(
        json.dumps({"format": fmt, "clips": entries}), encoding="utf-8"
    )
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    _fake_export(monkeypatch)
    monkeypatch.setattr(extend, "probe", _info)
    monkeypatch.setattr(extend, "export_clip_checked", clips.export_clip_checked)
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))
    return project


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


def _argv(cmd: str, project: Path, *extra: str) -> list[str]:
    if cmd == "clips":
        return ["clips", "p", "--segments", str(project / "seg.json"), *extra]
    clips_dir = str(project / "clips")
    return ["extend", clips_dir, "--seconds", "1", "--no-speech-aware", "--out", clips_dir, *extra]


def _only_original_folders(project: Path) -> None:
    assert sorted(p.name for p in project.iterdir()) == ["clips", "seg.json", "src.mp4"]


@pytest.mark.parametrize("exc", FAILURES, ids=["ffmpeg-error", "ctrl-c"])
def test_clips_force_that_fails_before_the_swap_keeps_the_old_clips(monkeypatch, home, exc):
    project = _setup(home, monkeypatch)
    before = _snapshot(project / "clips")
    calls = []

    def export(src, info, start, end, out, *_a, **_kw):
        calls.append(out)
        if len(calls) == 2:
            raise exc
        out.write_bytes(b"new")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    assert cli.main(_argv("clips", project, "--force")) == (
        130 if isinstance(exc, KeyboardInterrupt) else 1
    )
    assert len(calls) == 2
    assert _snapshot(project / "clips") == before
    _only_original_folders(project)


def _fail_nth_swap_replace(monkeypatch, project: Path, n: int, mode: str):
    """Fail the n-th distinct move in or out of the clips folder (retries of it included)."""
    clips_dir = project / "clips"
    seen: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def replace(self, target):
        target = Path(target)
        if {self.parent.name, target.parent.name} & {"clips", ".clips.cutpoint-bak"}:
            if (self, target) not in seen:
                seen.append((self, target))
            if seen[n - 1 : n] == [(self, target)]:
                if mode == "ctrl-c":
                    raise KeyboardInterrupt
                raise _locked(32)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    assert clips_dir.is_dir()
    return seen


@pytest.mark.parametrize("cmd", ["clips", "extend"])
@pytest.mark.parametrize("mode", ["lock", "ctrl-c"])
@pytest.mark.parametrize("n", [1, 3, 5, 8], ids=["first", "backing-up", "moving-in", "last"])
def test_failed_swap_puts_every_original_back(monkeypatch, home, caplog, cmd, mode, n):
    project = _setup(home, monkeypatch)
    before = _snapshot(project / "clips")
    seen = _fail_nth_swap_replace(monkeypatch, project, n, mode)
    rc = cli.main(_argv(cmd, project, "--force"))
    assert len(seen) >= n, "the swap never reached the failing move"
    if mode == "lock":
        assert rc == 1
        assert "open in another program" in caplog.text
    else:
        assert rc == 130
    assert _snapshot(project / "clips") == before
    _only_original_folders(project)


def _killed_mid_swap(monkeypatch, project: Path, cmd: str, n: int) -> dict[str, bytes]:
    """Run the command until the n-th move, then 'kill' it: no rollback or cleanup runs."""
    before = _snapshot(project / "clips")
    clips_dir = project / "clips"
    real_replace, real_unlink = Path.replace, Path.unlink
    moves = []
    dead = []

    def replace(self, target):
        if dead:
            raise _HardKill
        if {self.parent.name, Path(target).parent.name} & {"clips", ".clips.cutpoint-bak"}:
            moves.append(self)
            if len(moves) == n:
                dead.append(1)
                raise _HardKill
        return real_replace(self, target)

    def unlink(self, *a, **kw):
        if dead:
            raise _HardKill
        return real_unlink(self, *a, **kw)

    with monkeypatch.context() as m:  # the process is gone afterwards: real calls again
        m.setattr(Path, "replace", replace)
        m.setattr(Path, "unlink", unlink)
        with pytest.raises(_HardKill):
            cli.main(_argv(cmd, project, "--force"))
    assert clips_dir.is_dir() and _snapshot(clips_dir) != before
    return before


@pytest.mark.parametrize("cmd", ["clips", "extend"])
@pytest.mark.parametrize("n", [3, 6], ids=["while-backing-up", "while-moving-in"])
def test_next_run_after_a_kill_mid_swap_restores_the_originals(monkeypatch, home, cmd, n):
    project = _setup(home, monkeypatch)
    before = _killed_mid_swap(monkeypatch, project, cmd, n)
    assert (project / ".clips.cutpoint-bak").is_dir()
    # Without --force the run refuses (clips) or writes elsewhere (extend); either way it
    # starts by putting the originals back.
    argv = _argv(cmd, project)
    if cmd == "extend":
        argv[argv.index("--out") + 1] = str(project / "elsewhere")
    cli.main(argv)
    assert _snapshot(project / "clips") == before
    assert not (project / ".clips.cutpoint-bak").exists()
    assert not (project / ".clips.cutpoint-tmp").exists()


@pytest.mark.parametrize("cmd", ["clips", "extend"])
def test_backup_without_a_journal_means_the_swap_finished(monkeypatch, home, cmd):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    for n in NAMES:  # the finished new set
        (clips_dir / n).write_bytes(b"new")
    bak = project / ".clips.cutpoint-bak"
    bak.mkdir()
    (bak / "001_a.mp4").write_bytes(b"old")  # the leftover originals, journal already gone
    before = _snapshot(clips_dir)
    argv = _argv(cmd, project)
    if cmd == "extend":
        argv[argv.index("--out") + 1] = str(project / "elsewhere")
    cli.main(argv)
    assert _snapshot(clips_dir) == before
    assert not bak.exists()


@pytest.mark.parametrize("cmd", ["clips", "extend"])
def test_force_replaces_clips_and_keeps_the_users_other_files(monkeypatch, home, cmd):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    (clips_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    (clips_dir / "extra").mkdir()
    (clips_dir / "extra" / "a.txt").write_text("deep", encoding="utf-8")
    (clips_dir / "009_gone.mp4").write_bytes(b"stale")
    assert cli.main(_argv(cmd, project, "--force")) == 0
    assert (clips_dir / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert (clips_dir / "extra" / "a.txt").read_text(encoding="utf-8") == "deep"
    # clips --force drops clips the new set lacks; extend only rewrites what the manifest lists
    assert (clips_dir / "009_gone.mp4").exists() is (cmd == "extend")
    assert all((clips_dir / n).read_bytes() == b"new" for n in NAMES)
    manifest = (clips_dir / "manifest.json").read_text(encoding="utf-8")
    assert "cutpoint-tmp" not in manifest
    assert [c["file"] for c in json.loads(manifest)["clips"]] == NAMES
    _only_original_folders(project)


@pytest.mark.parametrize("cmd", ["clips", "extend"])
@pytest.mark.parametrize("journal", ["", "[]"], ids=["zero-length", "wrong-shape"])
def test_backup_with_a_corrupt_journal_is_kept_and_named(monkeypatch, home, caplog, cmd, journal):
    project = _setup(home, monkeypatch)
    bak = project / ".clips.cutpoint-bak"
    bak.mkdir()
    (bak / "001_a.mp4").write_bytes(b"precious original")
    (bak / ".cutpoint-swap.json").write_text(journal, encoding="utf-8")
    before = _snapshot(bak)
    argv = _argv(cmd, project)
    if cmd == "extend":
        argv[argv.index("--out") + 1] = str(project / "elsewhere")
    assert cli.main(argv) == 1
    assert ".clips.cutpoint-bak" in caplog.text
    assert _snapshot(bak) == before


def test_rollback_with_one_stuck_original_restores_the_rest_and_finishes_next_run(
    monkeypatch, home
):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    before = _snapshot(clips_dir)
    locked = {"001_a.mp4": None, "003_c.mp4": None}  # moving 003 in fails, then 001 is stuck
    real_replace = Path.replace

    def replace(self, target):
        stuck = (
            self.name == "003_c.mp4" and self.parent.name == ".clips.cutpoint-tmp",
            self.name == "001_a.mp4" and self.parent.name == ".clips.cutpoint-bak",
        )
        if locked and any(stuck):
            raise _locked(32)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    assert cli.main(_argv("clips", project, "--force")) == 1
    bak = project / ".clips.cutpoint-bak"
    assert (bak / "001_a.mp4").read_bytes() == before["001_a.mp4"]  # the stuck one
    assert (bak / ".cutpoint-swap.json").exists()
    for name in ("002_b.mp4", "003_c.mp4", "manifest.json"):  # the rest came back
        assert (clips_dir / name).read_bytes() == before[name]

    locked.clear()  # the lock is gone; no --force needed to recover
    cli.main(_argv("clips", project))
    assert _snapshot(clips_dir) == before
    _only_original_folders(project)


def test_clips_force_with_dot_as_out_inside_the_clips_folder(monkeypatch, home):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    monkeypatch.chdir(clips_dir)  # the CLI passes Path(".") on unchanged
    assert cli.main(_argv("clips", project, "--out", ".", "--force")) == 0
    assert all((clips_dir / n).read_bytes() == b"new" for n in NAMES)
    _only_original_folders(project)
