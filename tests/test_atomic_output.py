"""Exports appear under their final name only when complete."""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import hard_link, link_file, media_info, messages, remap
from cutpoint import assemble, cli, clips, encode, extend, ffmpeg_utils, planning
from cutpoint.ffmpeg_utils import FFmpegError


def _dies_midway(exc: BaseException):
    def run(args, **_kw):
        Path(args[-1]).write_bytes(b"truncated")  # ffmpeg had started writing
        raise exc

    return run


FAILURES = [FFmpegError("ffmpeg failed (exit 1)"), KeyboardInterrupt()]


def test_interrupted_clip_export_leaves_no_file(monkeypatch, tmp_path):
    """Any failure is cleaned up the same way (see test_gpu_encode for an ffmpeg error)."""
    exc = KeyboardInterrupt()
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(encode, "run", _dies_midway(exc))
    src = tmp_path / "src.mp4"
    with pytest.raises(type(exc)):
        encode.export_clip(
            src,
            media_info(src),
            0.0,
            2.0,
            tmp_path / "001.mp4",
            encode.OutputFormat(1920, 1080, 30),
            False,
        )
    assert list(tmp_path.iterdir()) == []


def test_interrupted_assemble_leaves_no_final_video(monkeypatch, tmp_path):
    exc = KeyboardInterrupt()
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "001.mp4").write_bytes(b"x")
    monkeypatch.setattr(assemble, "probe", media_info)
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


def test_stop_signal_cancels_a_command_like_ctrl_c(monkeypatch):
    """A UI cancel (SIGTERM) or Windows Ctrl+Break (SIGBREAK) unwinds as exit 130.

    The caller's handler is left alone afterwards.
    """
    stop = signal.SIGBREAK if os.name == "nt" else signal.SIGTERM

    def cancelled(_args):
        signal.raise_signal(stop)
        return 0

    monkeypatch.setattr(cli, "_cmd_paths", cancelled)
    before = signal.getsignal(stop)

    assert cli.main(["paths"]) == 130
    assert signal.getsignal(stop) is before


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
        "Popen",
        lambda cmd, **_kw: SimpleNamespace(communicate=lambda timeout: ("", "boom"), returncode=rc),
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
    """#109: a failed fresh run leaves no folder it created and no partial clips (#160: the
    clips are still staged, so the first clip is not left behind when the second fails)."""
    project, seg = _project(home)
    out = project / "clips"
    if preexisting:
        out.mkdir()

    def leftovers() -> list[str]:
        return sorted(p.name for p in out.iterdir()) if out.exists() else []

    def second_clip_fails(src, info, start, end, dest, *_a, **_kw):
        if dest.name.startswith("001"):
            dest.write_bytes(b"new")
        else:
            raise FFmpegError("ffmpeg failed (exit 1)")

    segments = json.loads(seg.read_text(encoding="utf-8"))
    seg.write_text(json.dumps([*segments, {**segments[0], "label": "b"}]), encoding="utf-8")
    monkeypatch.setattr(clips, "probe", media_info)
    monkeypatch.setattr(clips, "export_clip_checked", second_clip_fails)
    with pytest.raises(FFmpegError):
        clips.export_clips(project, segments=seg)
    assert leftovers() == []
    assert out.is_dir() is preexisting
    assert sorted(p.name for p in project.iterdir() if p.name.startswith(".")) == []

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

    monkeypatch.setattr(clips, "probe", media_info)
    monkeypatch.setattr(clips, "export_clip_checked", export)


@pytest.mark.parametrize("winerror", [32, 5])
@pytest.mark.usefixtures("no_sleep")
def test_clips_force_over_a_locked_clip_says_who_to_close(monkeypatch, home, caplog, winerror):
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


@pytest.mark.usefixtures("no_sleep")
def test_clips_force_retries_a_lock_that_clears(monkeypatch, home):
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


@pytest.mark.usefixtures("no_sleep")
def test_locked_final_video_names_the_file(monkeypatch, tmp_path):
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
@pytest.mark.usefixtures("no_sleep")
def test_non_lock_permission_error_is_reraised_unchanged(monkeypatch, tmp_path, winerror):
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
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
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
    _fake_export(monkeypatch)
    monkeypatch.setattr(extend, "probe", media_info)
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


def _fail_nth_swap_replace(monkeypatch, project: Path, n: int, mode: str, folder: str = "clips"):
    """Fail the n-th distinct move in or out of the folder (retries of it included)."""
    clips_dir = project / folder
    seen: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def replace(self, target):
        target = Path(target)
        if {self.parent.name, target.parent.name} & {folder, f".{folder}.cutpoint-bak"}:
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


@pytest.mark.parametrize("mode", ["lock", "ctrl-c"])
@pytest.mark.parametrize(
    ("cmd", "n"),
    [("clips", n) for n in (1, 3, 5, 8)] + [("extend", 5)],  # extend shares swap_in with clips
    ids=["clips-first", "clips-backing-up", "clips-moving-in", "clips-last", "extend-moving-in"],
)
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


@pytest.mark.parametrize(
    ("fail", "n"),
    [("export-ctrl-c", 0), ("lock", 4)],
)
def test_extend_out_force_into_a_non_empty_folder_stays_unchanged_when_it_fails(
    monkeypatch, home, fail, n
):
    """#147: extend --out DIR --force swaps like in place, so a failure leaves DIR as it was."""
    project = _setup(home, monkeypatch)
    out = project / "out"
    shutil.copytree(project / "clips", out)
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    before = _snapshot(out)
    if fail == "export-ctrl-c":
        calls = []

        def export(src, info, start, end, dst, *_a, **_kw):
            calls.append(dst)
            if len(calls) == 2:
                raise KeyboardInterrupt
            dst.write_bytes(b"new")

        monkeypatch.setattr(extend, "export_clip_checked", export)
        seen = None
    else:
        seen = _fail_nth_swap_replace(monkeypatch, project, n, fail, folder="out")
    clips_dir = str(project / "clips")
    argv = [
        "extend",
        clips_dir,
        "--seconds",
        "1",
        "--no-speech-aware",
        "--out",
        str(out),
        "--force",
    ]
    assert cli.main(argv) == (1 if fail == "lock" else 130)
    if seen is not None:
        assert len(seen) >= n, "the swap never reached the failing move"
    assert _snapshot(out) == before
    assert sorted(p.name for p in project.iterdir()) == ["clips", "out", "seg.json", "src.mp4"]


def test_extend_out_finishes_a_killed_earlier_replace_before_checking_it_is_empty(
    monkeypatch, home
):
    """#147: a hard-killed `extend --out DIR --force` left DIR empty and its originals in a backup."""
    project = _setup(home, monkeypatch)
    out = project / "out"
    out.mkdir()
    bak = project / ".out.cutpoint-bak"
    bak.mkdir()
    (bak / "001_a.mp4").write_bytes(b"old out")
    journal = {"old": ["001_a.mp4"], "new": ["001_a.mp4"], "created": False}
    (bak / ".cutpoint-swap.json").write_text(json.dumps(journal), encoding="utf-8")
    argv = ["extend", str(project / "clips"), "--seconds", "1", "--no-speech-aware"]
    assert cli.main([*argv, "--out", str(out)]) == 1  # restored, so not empty: needs --force
    assert (out / "001_a.mp4").read_bytes() == b"old out"
    assert not bak.exists()


@pytest.mark.parametrize("kind", ["mount point", "drive root"])
def test_extend_out_force_refuses_a_mount_point_or_root_before_exporting(
    monkeypatch, home, caplog, kind
):
    """#147: the staging and backup folders sit next to --out, so a mount point or a drive root
    (no name, no parent to hold them) cannot be swapped. Only ``out`` is made to look like one."""
    project = _setup(home, monkeypatch)
    out = project / "out"
    shutil.copytree(project / "clips", out)
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    before = _snapshot(out)
    real_is_mount, real_resolve = Path.is_mount, Path.resolve
    if kind == "mount point":
        monkeypatch.setattr(
            Path, "is_mount", lambda self: self.resolve() == out.resolve() or real_is_mount(self)
        )
    else:
        # A nameless path stands in for the root; nothing is ever opened at the real root.
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda self, *a, **kw: (
                Path(out.anchor) if self == out else real_resolve(self, *a, **kw)
            ),
        )

    def export(*_a, **_kw):
        raise AssertionError("exported before the mount point was refused")

    monkeypatch.setattr(extend, "export_clip_checked", export)
    argv = ["extend", str(project / "clips"), "--seconds", "1", "--no-speech-aware"]
    assert cli.main([*argv, "--out", str(out), "--force"]) == 1
    assert kind in caplog.text
    assert _snapshot(out) == before
    assert sorted(p.name for p in project.iterdir()) == ["clips", "out", "seg.json", "src.mp4"]


@pytest.mark.parametrize("cmd", ["clips", "extend"])
@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("drive root", "is a drive root"),
        ("mount point", "is a mount point"),
        ("volume mount", "is a mount point"),
        ("bind mount", "is on another file system than its parent folder"),
    ],
)
def test_clips_force_and_in_place_extend_refuse_a_root_or_mount_before_exporting(
    monkeypatch, home, caplog, cmd, kind, message
):
    """#160: the staging folder sits next to the clips folder, so a drive root (no name, crashed
    with "empty name"), a mount point or a bind mount (a rename into it fails with EXDEV) is
    refused with one ERROR line before any export, and nothing is left behind."""
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    (clips_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    before = _snapshot(clips_dir)
    real_is_mount, real_resolve, real_rename = Path.is_mount, Path.resolve, Path.rename
    if kind == "mount point":
        monkeypatch.setattr(
            Path,
            "is_mount",
            lambda self: self.resolve() == clips_dir.resolve() or real_is_mount(self),
        )
    elif kind in ("drive root", "volume mount"):
        # A nameless path stands in for the root; nothing is ever opened at the real root.
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda self, *a, **kw: (
                Path(clips_dir.anchor) if self == clips_dir else real_resolve(self, *a, **kw)
            ),
        )
        if kind == "volume mount":  # #185: a volume mounted into a folder resolves to a root
            monkeypatch.setattr(
                Path, "is_mount", lambda self: self == clips_dir or real_is_mount(self)
            )
    else:

        def rename(self, target):
            if self.name == ffmpeg_utils._SWAP_PROBE:
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", rename)

    def export(*_a, **_kw):
        raise AssertionError("exported before the folder was refused")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    monkeypatch.setattr(extend, "export_clip_checked", export)
    assert cli.main(_argv(cmd, project, "--force")) == 1
    assert message in _errors(caplog)
    assert _snapshot(clips_dir) == before
    _only_original_folders(project)


@pytest.mark.parametrize(
    ("folder", "is_mount", "message"),
    [
        ("root", True, "is a drive root"),
        pytest.param(
            "unc", True, "is a drive root",
            marks=pytest.mark.skipif(os.name != "nt", reason="UNC paths"),
        ),
    ],
)  # fmt: skip
def test_swap_target_that_resolves_to_a_nameless_root(
    monkeypatch, tmp_path, folder, is_mount, message
):
    """#185: only a folder that is a mount point itself is called one; a real root or a link
    that lands on a root keeps the drive-root message. The error names the folder as given."""
    dst = {
        "root": Path(tmp_path.anchor),
        "unc": Path(r"\\srv\share"),
    }[folder]
    root = Path(tmp_path.anchor)
    monkeypatch.setattr(Path, "resolve", lambda self, *a, **kw: root if self == dst else self)
    monkeypatch.setattr(Path, "is_mount", lambda self: is_mount and self == dst)
    with pytest.raises(ValueError, match=message) as err:
        ffmpeg_utils.check_swap_target(dst)
    assert str(dst) in str(err.value)


@pytest.mark.parametrize("kind", ["drive root", "mount point", "bind mount"])
def test_clips_into_an_empty_root_or_mount_point_exports_directly(monkeypatch, home, caplog, kind):
    """#160: nothing is replaced in an empty folder, so no staging folder is needed and
    ``clips`` without --force works there as before."""
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    shutil.rmtree(clips_dir)
    clips_dir.mkdir()
    real_is_mount, real_resolve = Path.is_mount, Path.resolve
    if kind == "mount point":
        monkeypatch.setattr(
            Path,
            "is_mount",
            lambda self: self.resolve() == clips_dir.resolve() or real_is_mount(self),
        )
    elif kind == "bind mount":
        real_rename = Path.rename

        def rename(self, target):
            if self.name == ffmpeg_utils._SWAP_PROBE:
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", rename)
    else:
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda self, *a, **kw: (
                Path(clips_dir.anchor) if self == clips_dir else real_resolve(self, *a, **kw)
            ),
        )
    assert cli.main(_argv("clips", project)) == 0
    assert all((clips_dir / n).read_bytes() == b"new" for n in NAMES)
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


def test_backup_without_a_journal_means_the_swap_finished(monkeypatch, home):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    for n in NAMES:  # the finished new set
        (clips_dir / n).write_bytes(b"new")
    bak = project / ".clips.cutpoint-bak"
    bak.mkdir()
    (bak / "001_a.mp4").write_bytes(b"old")  # the leftover originals, journal already gone
    before = _snapshot(clips_dir)
    cli.main(_argv("clips", project))
    assert _snapshot(clips_dir) == before
    assert not bak.exists()


@pytest.mark.parametrize("cmd", ["clips", "extend"])
def test_force_replaces_clips_and_keeps_the_users_other_files(monkeypatch, home, caplog, cmd):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    (clips_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    (clips_dir / "extra").mkdir()
    (clips_dir / "extra" / "a.txt").write_text("deep", encoding="utf-8")
    (clips_dir / "009_gone.mp4").write_bytes(b"stale")
    assert cli.main(_argv(cmd, project, "--force")) == 0
    assert (clips_dir / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert (clips_dir / "extra" / "a.txt").read_text(encoding="utf-8") == "deep"
    # not in the old manifest: not made by cutpoint, so neither command replaces it
    assert (clips_dir / "009_gone.mp4").read_bytes() == b"stale"
    assert all((clips_dir / n).read_bytes() == b"new" for n in NAMES)
    manifest = (clips_dir / "manifest.json").read_text(encoding="utf-8")
    assert "cutpoint-tmp" not in manifest
    assert [c["file"] for c in json.loads(manifest)["clips"]] == NAMES
    _only_original_folders(project)
    # extend --out DIR (not the input folder) keeps the user's files too, and says so
    if cmd != "extend":
        return
    out = project / "out"
    shutil.copytree(clips_dir, out, ignore=shutil.ignore_patterns("extra"))
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    (out / "009_gone.mp4").write_bytes(b"stale")
    argv = ["extend", str(clips_dir), "--seconds", "1", "--no-speech-aware", "--out", str(out)]
    caplog.clear()
    assert cli.main([*argv, "--force"]) == 0
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert (out / "009_gone.mp4").read_bytes() == b"stale"
    assert "009_gone.mp4" in caplog.text
    assert all((out / n).read_bytes() == b"new" for n in NAMES)
    out_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [c["file"] for c in out_manifest["clips"]] == NAMES


@pytest.mark.parametrize("journal", ["", "[]"], ids=["zero-length", "wrong-shape"])
def test_backup_with_a_corrupt_journal_is_kept_and_named(monkeypatch, home, caplog, journal):
    project = _setup(home, monkeypatch)
    bak = project / ".clips.cutpoint-bak"
    bak.mkdir()
    (bak / "001_a.mp4").write_bytes(b"precious original")
    (bak / ".cutpoint-swap.json").write_text(journal, encoding="utf-8")
    before = _snapshot(bak)
    assert cli.main(_argv("clips", project)) == 1
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


# --- --force replaces only what an earlier run made; a source is never replaced ---


def _warnings(caplog) -> str:
    return "\n".join(messages(caplog))


def _errors(caplog) -> str:
    return "\n".join(messages(caplog, "ERROR"))


def _segments(path: Path, src: Path, labels: str = "a") -> None:
    path.write_text(
        json.dumps(
            [{"source": str(src), "start": i * 5.0, "end": i * 5.0 + 2, "label": c}
             for i, c in enumerate(labels)]
        ),
        encoding="utf-8",
    )  # fmt: skip


def test_clips_force_keeps_a_source_and_other_files_in_the_output_folder(monkeypatch, home, caplog):
    out = home / "projects" / "p" / "clips"
    out.mkdir(parents=True)
    src = out / "source.mp4"
    src.write_bytes(b"precious source")
    (out / "readme.txt").write_text("hi", encoding="utf-8")
    _segments(home / "seg.json", src)
    _fake_export(monkeypatch)
    argv = ["clips", "p", "--segments", str(home / "seg.json"), "--out", str(out), "--force"]
    assert cli.main(argv) == 0
    assert src.read_bytes() == b"precious source"
    assert (out / "readme.txt").read_text(encoding="utf-8") == "hi"
    assert (out / "001_a.mp4").read_bytes() == b"new"
    assert "source.mp4" in _warnings(caplog)


@pytest.mark.parametrize("kind", ["plain", "symlink"])
def test_clips_force_refuses_a_new_clip_name_that_is_a_source(monkeypatch, home, caplog, kind):
    project, seg = _project(home)
    out = project / "clips"
    out.mkdir()
    if kind == "plain":
        src = out / "001_a.mp4"
        src.write_bytes(b"precious source")
    else:
        src = project / "real.mp4"
        src.write_bytes(b"precious source")
        link_file(out / "001_a.mp4", src)
    _segments(seg, src)
    (out / "notes.txt").write_text("keep", encoding="utf-8")
    exported = []
    _fake_export(monkeypatch)
    monkeypatch.setattr(clips, "export_clip_checked", lambda *a, **_kw: exported.append(a))
    before = sorted(p.name for p in out.iterdir())
    assert cli.main(["clips", "p", "--segments", str(seg), "--force"]) == 1
    assert "001_a.mp4" in _errors(caplog)
    assert not exported
    assert src.read_bytes() == b"precious source"
    assert sorted(p.name for p in out.iterdir()) == before
    assert sorted(p.name for p in project.iterdir() if p.name.startswith(".")) == []


def _write_old_set(project: Path, manifest: str | None) -> Path:
    out = project / "clips"
    for n in (*NAMES, "009_user.mp4", ".004_x.part.mp4"):
        (out / n).write_bytes(b"old")
    if manifest is None:
        (out / "manifest.json").unlink()
    else:
        (out / "manifest.json").write_text(manifest, encoding="utf-8")
    return out


def test_clips_force_replaces_only_clips_of_the_old_manifest(monkeypatch, home, caplog):
    project = _setup(home, monkeypatch)
    _segments(project / "seg.json", project / "src.mp4")  # one clip: 002 and 003 are dropped
    out = _write_old_set(project, (project / "clips" / "manifest.json").read_text("utf-8"))
    assert cli.main(_argv("clips", project, "--force")) == 0
    assert sorted(p.name for p in out.glob("*.mp4") if not p.name.startswith(".")) == [
        "001_a.mp4",
        "009_user.mp4",
    ]
    assert (out / "001_a.mp4").read_bytes() == b"new"
    assert (out / "009_user.mp4").read_bytes() == b"old"
    assert not (out / ".004_x.part.mp4").exists()
    assert "009_user.mp4" in _warnings(caplog)


@pytest.mark.parametrize("manifest", [None, "{not json"], ids=["missing", "unreadable"])
def test_clips_force_without_a_readable_manifest_replaces_only_the_new_names(
    monkeypatch, home, caplog, manifest
):
    project = _setup(home, monkeypatch)
    _segments(project / "seg.json", project / "src.mp4")
    out = _write_old_set(project, manifest)
    assert cli.main(_argv("clips", project, "--force")) == 0
    assert (out / "001_a.mp4").read_bytes() == b"new"
    for n in ("002_b.mp4", "003_c.mp4", "009_user.mp4"):
        assert (out / n).read_bytes() == b"old"
    assert "manifest.json" in _warnings(caplog)


def _point_manifest_at(clips_dir: Path, name: str, source: Path) -> None:
    path = clips_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data["clips"]:
        if e["file"] == name:
            e["source"] = str(source)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("where", ["in-place", "out-is-source-folder"])
def test_extend_refuses_to_overwrite_a_source_before_exporting(monkeypatch, home, caplog, where):
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    if where == "in-place":
        src, out = clips_dir / "001_a.mp4", clips_dir
    else:
        out = project / "footage"
        out.mkdir()
        src = out / "001_a.mp4"
        src.write_bytes(b"precious source")
    _point_manifest_at(clips_dir, "001_a.mp4", src)
    exported = []
    monkeypatch.setattr(extend, "export_clip_checked", lambda *a, **_kw: exported.append(a))
    before_src, before_out = src.read_bytes(), sorted(p.name for p in out.iterdir())
    argv = ["extend", str(clips_dir), "--seconds", "1", "--no-speech-aware"]
    assert cli.main([*argv, "--out", str(out), "--force"]) == 1
    assert "001_a.mp4" in _errors(caplog)
    assert not exported
    assert src.read_bytes() == before_src
    assert sorted(p.name for p in out.iterdir()) == before_out
    assert not (project / ".clips.cutpoint-tmp").exists()


# --- a leftover that cannot be removed stops the run with a hint ---


@pytest.mark.parametrize("leftover", [".clips.cutpoint-bak", ".clips.cutpoint-tmp"])
def test_locked_leftover_folder_stops_the_run_until_it_is_released(
    monkeypatch, home, caplog, leftover
):
    project = _setup(home, monkeypatch)
    stale = project / leftover
    stale.mkdir()
    (stale / "001_a.mp4").write_bytes(b"old")  # no journal: nothing in it is needed
    locked = [True]
    real_rmtree = shutil.rmtree

    def rmtree(path, *a, **kw):
        if locked and Path(path) == stale:
            raise _locked(32)
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    argv = _argv("clips", project, "--force")
    assert cli.main(argv) == 1
    assert "open in another program" in caplog.text
    assert leftover in caplog.text
    assert stale.exists()

    locked.clear()
    assert cli.main(argv) == 0
    assert all((project / "clips" / n).read_bytes() == b"new" for n in NAMES)
    _only_original_folders(project)


# --- a manifest-listed clip that this run reads as a source is kept, however it is reached ---


@pytest.mark.parametrize("kind", ["old-clip", "hard-link"])
def test_clips_force_keeps_an_old_clip_that_is_this_runs_source(monkeypatch, home, caplog, kind):
    project = _setup(home, monkeypatch)
    out = project / "clips"
    kept = out / "003_c.mp4"
    if kind == "old-clip":
        source = kept
        expected = b"old 003_c.mp4"
    else:
        kept.unlink()
        source = project / "src.mp4"
        hard_link(kept, source)
        expected = source.read_bytes()
    _segments(project / "seg.json", source)
    assert cli.main(_argv("clips", project, "--force")) == 0
    assert kept.read_bytes() == expected
    assert not (out / "002_b.mp4").exists()
    warned = _warnings(caplog)
    assert "reads as a source" in warned
    assert "003_c.mp4" in warned


def test_clips_force_keeps_a_clip_reached_through_source_remap(monkeypatch, home, caplog, tmp_path):
    project = _setup(home, monkeypatch)
    out = project / "clips"
    gone = tmp_path / "gone"
    remap(home, monkeypatch, gone, out)
    _segments(project / "seg.json", gone / "003_c.mp4")
    assert cli.main(_argv("clips", project, "--force")) == 0
    assert (out / "003_c.mp4").read_bytes() == b"old 003_c.mp4"
    assert "reads as a source" in _warnings(caplog)


def test_clips_force_refuses_a_new_clip_name_reached_through_source_remap(
    monkeypatch, home, tmp_path
):
    project = _setup(home, monkeypatch)
    out = project / "clips"
    gone = tmp_path / "gone"
    remap(home, monkeypatch, gone, out)
    _segments(project / "seg.json", gone / "001_a.mp4")
    before = _snapshot(out)
    assert cli.main(_argv("clips", project, "--force")) == 1
    assert _snapshot(out) == before


@pytest.mark.parametrize(("winerror", "read_only_hint"), [(32, False), (5, True)])
@pytest.mark.usefixtures("no_sleep")
def test_lock_hint_mentions_read_only_only_for_access_denied(
    monkeypatch, tmp_path, winerror, read_only_hint
):

    def op():
        raise _locked(winerror)

    with pytest.raises(PermissionError, match="open in another program") as exc:
        ffmpeg_utils.retry_locked(op, tmp_path / "x.json", "write")
    assert ("read-only" in str(exc.value)) is read_only_hint


# --- staged_dir(direct_if_empty) and retry_locked(locked=) ---


def test_failed_direct_export_into_an_empty_mount_keeps_the_clips_already_written(
    monkeypatch, home, caplog
):
    """#160: an empty folder that cannot be swapped is exported into directly; a failing
    export leaves the earlier clips (nothing was replaced) and no staging folder or warning."""
    project = _setup(home, monkeypatch)
    clips_dir = project / "clips"
    shutil.rmtree(clips_dir)
    clips_dir.mkdir()
    real_rename = Path.rename

    def rename(self, target):
        if self.name == ffmpeg_utils._SWAP_PROBE:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    calls = []

    def export(src, info, start, end, out, *_a, **_kw):
        calls.append(out)
        if len(calls) == 2:
            raise FFmpegError("ffmpeg failed (exit 1)")
        out.write_bytes(b"new")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    assert cli.main(_argv("clips", project)) == 1
    assert len(calls) == 2
    assert [p.name for p in clips_dir.glob("001_*")] != []
    assert not (project / ".clips.cutpoint-tmp").exists()
    assert "Could not remove" not in caplog.text


@pytest.mark.usefixtures("no_sleep")
def test_retry_locked_predicate_picks_what_is_retried_and_reraises_the_last_unchanged(
    monkeypatch, tmp_path
):
    errs = [_locked(32) for _ in range(ffmpeg_utils.REPLACE_ATTEMPTS)]
    calls = []

    def op():
        calls.append(1)
        raise errs[len(calls) - 1]

    with pytest.raises(OSError) as exc:
        ffmpeg_utils.retry_locked(op, tmp_path / "x", "remove", lambda e: True)
    assert exc.value is errs[-1]  # not reworded, though winerror 32 would be
    assert len(calls) == ffmpeg_utils.REPLACE_ATTEMPTS


@pytest.mark.usefixtures("no_sleep")
def test_retry_locked_predicate_rejecting_an_error_does_not_retry_it(monkeypatch, tmp_path):
    calls = []

    def op():
        calls.append(1)
        raise PermissionError(13, "denied")  # retried by default, rejected by this predicate

    with pytest.raises(PermissionError, match="denied"):
        ffmpeg_utils.retry_locked(op, tmp_path / "x", "remove", lambda e: False)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            _locked(32),
            "Cannot remove {p}, why: it is open in another program (a player, an editor or a "
            "virus scanner). Close it, or delete the folder, and re-run. (The process cannot "
            "access the file)",
        ),
        (
            OSError(errno.EIO, "Input/output error"),
            "Cannot remove {p}, why: Input/output error. Delete the folder and re-run.",
        ),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_remove_leftover_messages(monkeypatch, tmp_path, error, message):

    def rmtree(*_a, **_kw):
        raise error

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    expected_type = PermissionError if isinstance(error, PermissionError) else OSError
    with pytest.raises(expected_type) as exc:
        ffmpeg_utils._remove_leftover(tmp_path, "why")
    assert str(exc.value) == message.format(p=tmp_path)
    assert exc.value.__cause__ is error


def test_extract_audio_leaves_no_temp_wav_when_it_fails_or_is_interrupted_midway(
    monkeypatch, tmp_path
):
    dst = tmp_path / "cache" / "a.wav"
    for exc in (FFmpegError("ffmpeg failed"), KeyboardInterrupt()):

        def run(args, *_a, exc=exc, **_kw):
            Path(args[-1]).write_bytes(b"partial")  # ffmpeg had started writing
            raise exc

        monkeypatch.setattr(ffmpeg_utils, "run_progress", run)
        with pytest.raises(type(exc)) as err:
            ffmpeg_utils.extract_audio(tmp_path / "a.mp4", dst, media_info(tmp_path / "a.mp4"))
        assert err.value is exc
        assert list(dst.parent.iterdir()) == []


# --- #199: an interrupted ffmpeg exits before its partial output is removed ---


class _Child:
    """Stand-in for the ffmpeg ``Popen`` of ``run`` and ``run_progress``; it never sleeps.

    ``communicate`` or iterating ``stdout`` raises ``raises`` once (then it is running). The
    child exits ``exits_after`` s into a ``wait`` (None: only when killed, never when
    ``unkillable``); ``on_exit`` runs as it exits. ``calls`` records wait/kill/close.
    """

    def __init__(self, raises=None, exits_after=0.0, unkillable=False, on_exit=None) -> None:
        self.raises, self.exits_after = raises, exits_after
        self.unkillable, self.on_exit = unkillable, on_exit
        self.returncode = None
        self.calls: list[tuple] = []
        self.stdout = self  # run_progress reads and closes it

    def _raise_once(self) -> None:
        if self.raises is not None:
            err, self.raises = self.raises, None
            raise err

    def __iter__(self):
        self._raise_once()
        return iter(["out_time_us=1000000\n"])

    def close(self) -> None:
        self.calls.append(("close",))

    def communicate(self, timeout=None):
        self._raise_once()
        self.wait()
        return "", ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.returncode is None:
            if self.exits_after is None or (timeout is not None and timeout < self.exits_after):
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            self._exit(0)
        return self.returncode

    def kill(self) -> None:
        self.calls.append(("kill",))
        if not self.unkillable:
            self._exit(-9)

    def _exit(self, code: int) -> None:
        self.returncode = code
        if self.on_exit is not None:
            self.on_exit()


@pytest.fixture
def child(monkeypatch):
    """Every ffmpeg started is ``state.proc``; its argv is ``state.cmd``."""
    state = SimpleNamespace(proc=_Child(), cmd=None)

    def popen(cmd, **_kw):
        state.cmd = cmd
        return state.proc

    monkeypatch.setattr(ffmpeg_utils, "find_tool", lambda name: name)
    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", popen)
    return state


_RUNNERS = {
    "run": lambda: ffmpeg_utils.run(["ffmpeg", "-i", "x.mp4", "out.mp4"]),
    "run_progress": lambda: ffmpeg_utils.run_progress(["ffmpeg", "-i", "x.mp4", "o.mp4"], 9, "t"),
}
_SOFT = ("wait", ffmpeg_utils.SOFT_STOP_S)


def _waits_and_kills(proc: _Child) -> list[tuple]:
    return [c for c in proc.calls if c[0] != "close"]


@pytest.mark.parametrize("runner", _RUNNERS)
def test_interrupted_ffmpeg_is_waited_for_and_the_interrupt_reraised(child, runner):
    """It got the Ctrl+C too and exits within SOFT_STOP_S: no kill."""
    exc = KeyboardInterrupt()
    child.proc = _Child(raises=exc, exits_after=ffmpeg_utils.SOFT_STOP_S / 2)
    with pytest.raises(KeyboardInterrupt) as err:
        _RUNNERS[runner]()
    assert err.value is exc
    assert _waits_and_kills(child.proc) == [_SOFT]
    assert child.proc.returncode == 0


@pytest.mark.parametrize("runner", _RUNNERS)
@pytest.mark.parametrize("unkillable", [False, True], ids=["killed", "unkillable"])
def test_interrupted_ffmpeg_still_running_after_soft_stop_is_killed(child, runner, unkillable):
    """Still running after SOFT_STOP_S: killed and waited for again, bounded both times."""
    exc = KeyboardInterrupt()
    child.proc = _Child(raises=exc, exits_after=None, unkillable=unkillable)
    with pytest.raises(KeyboardInterrupt) as err:
        _RUNNERS[runner]()
    assert err.value is exc
    assert _waits_and_kills(child.proc) == [_SOFT, ("kill",), _SOFT]
    assert (child.proc.returncode is None) is unkillable


@pytest.mark.parametrize("runner", _RUNNERS)
def test_ffmpeg_is_killed_at_once_on_other_errors(child, runner):
    """Not an interrupt: the child got no signal, so it is killed without the soft wait."""
    exc = MemoryError()
    child.proc = _Child(raises=exc, exits_after=None)
    with pytest.raises(MemoryError) as err:
        _RUNNERS[runner]()
    assert err.value is exc
    assert _waits_and_kills(child.proc) == [("kill",), _SOFT]


def test_run_progress_stops_reading_progress_before_waiting(child):
    child.proc = _Child(raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        _RUNNERS["run_progress"]()
    assert child.proc.calls[0] == ("close",)


def test_interrupted_extract_audio_removes_its_temp_wav_after_ffmpeg_exits(
    monkeypatch, tmp_path, child
):
    """Like Windows: the .tmp.wav cannot be removed while ffmpeg is running."""
    dst = tmp_path / "cache" / "a.wav"
    child.proc = _Child(
        raises=KeyboardInterrupt(),
        exits_after=1.0,
        on_exit=lambda: Path(child.cmd[-1]).write_bytes(b"finalized"),
    )
    unlink = Path.unlink

    def locked_unlink(self, *a, **kw):
        if child.proc.returncode is None:
            raise PermissionError(errno.EACCES, "in use by ffmpeg", str(self))
        unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    dst.parent.mkdir()
    (dst.parent / "a.tmp.wav").write_bytes(b"partial")  # ffmpeg had started writing
    with pytest.raises(KeyboardInterrupt):
        ffmpeg_utils.extract_audio(tmp_path / "a.mp4", dst, media_info(tmp_path / "a.mp4"))
    assert child.cmd[-1].endswith(".tmp.wav")
    assert list(dst.parent.iterdir()) == []
