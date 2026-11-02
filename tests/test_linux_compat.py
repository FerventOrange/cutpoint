"""Platform behaviour: path keys, Linux mounts, tool lookup, the sh wrapper.

``sys.platform`` is monkeypatched wherever the code branches on it, so every test is
valid on both Linux and Windows CI. Tests that need real POSIX features are skipped on Windows.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from conftest import messages, remap, set_config, write
from cutpoint import (
    assemble,
    cache,
    config,
    crash,
    errors,
    ffmpeg_utils,
    procs,
    stage,
    telemetry,
    transcribe,
)
from cutpoint.pano import stitch, x3

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX file semantics")


@pytest.fixture
def plat(monkeypatch: pytest.MonkeyPatch):
    """Call ``plat("win32" | "linux")`` to pretend to run there."""

    def set_platform(name: str) -> None:
        monkeypatch.setattr(sys, "platform", name)

    return set_platform


# -- config: path keys ------------------------------------------------------------------


def test_norm_path_windows_is_lowercase_backslash(plat):
    plat("win32")
    assert config.norm_path("D:/Videos/Trip/") == "d:\\videos\\trip"
    assert config.norm_path("D:\\Videos\\Trip") == "d:\\videos\\trip"
    assert config.path_key("D:\\Clip.MP4") == "d:\\clip.mp4"


def test_norm_path_posix_keeps_case_and_slashes(plat, monkeypatch):
    plat("linux")
    monkeypatch.setattr(config, "folds_case", lambda _f: False)  # / folds on macOS
    assert config.norm_path("/Data/Trip/") == "/Data/Trip"
    assert config.path_key("/Data/Clip.MP4") == "/Data/Clip.MP4"


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("C:\\a\\b.mp4", "b.mp4"),
        ("a\\b.mp4", "b.mp4"),
        ("/x/y.mp4", "y.mp4"),
    ],
)
def test_source_path_parses_windows_and_posix_spellings_on_linux(plat, text, name):
    plat("linux")
    assert config.source_path(text).name == name
    assert config.source_path(text).stem == name.removesuffix(".mp4")


@pytest.mark.parametrize("arg", ["projects/foo", "projects\\foo", "foo"])
def test_resolve_project_path_accepts_both_separators_and_bare_names(
    home, tmp_path, monkeypatch, arg
):
    monkeypatch.chdir(tmp_path)
    assert config.resolve_project_path(arg) == config.projects_dir() / "foo"


def test_resolve_project_path_keeps_a_sub_folder_with_backslashes(home, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = config.resolve_project_path("projects\\foo\\clips")
    assert got == config.projects_dir() / "foo" / "clips"


def test_canonical_source_windows_is_case_insensitive_with_backslashes(home, plat, monkeypatch):
    plat("win32")
    remap(home, monkeypatch, "D:\\Footage", "E:\\Footage-copy")
    assert config.canonical_source("e:/FOOTAGE-copy/Sub/x.mp4") == "D:\\Footage\\Sub\\x.mp4"
    assert config.canonical_source("D:\\Other\\x.mp4") == "D:\\Other\\x.mp4"


def test_canonical_source_posix_is_case_sensitive(home, plat, monkeypatch):
    plat("linux")
    remap(home, monkeypatch, "/data/Footage", "/mnt/copy")
    assert config.canonical_source("/mnt/copy/Sub/x.mp4") == "/data/Footage/Sub/x.mp4"
    assert config.canonical_source("/mnt/COPY/Sub/x.mp4") == "/mnt/COPY/Sub/x.mp4"
    assert config.canonical_source("/mnt/copy2/x.mp4") == "/mnt/copy2/x.mp4"  # not a prefix


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not block this user",
)
def test_remap_source_uses_the_original_when_the_target_cannot_be_checked(
    home, tmp_path, monkeypatch
):
    vault = tmp_path / "vault"
    vault.mkdir()
    remap(home, monkeypatch, "/data/Footage", str(vault / "copy"))
    vault.chmod(0)
    try:
        assert config.remap_source("/data/Footage/x.mp4") == Path("/data/Footage/x.mp4")
    finally:
        vault.chmod(0o755)


# -- crash: network mounts -----------------------------------------------------------------

MOUNTS = r"""sysfs /sys sysfs rw 0 0
/dev/sda1 / ext4 rw 0 0
nas:/export /nas nfs4 rw 0 0
//srv/share /nas/local ext4 rw 0 0
//srv/my\040share /mnt/my\040share cifs rw 0 0
/dev/sdb1 /media/disk ext4 rw 0 0
gvfsd-fuse /run/user/1000/gvfs fuse.gvfsd-fuse rw 0 0
rclone: /mnt/cloud fuse.rclone rw 0 0
broken
"""


@pytest.fixture
def proc_mounts(monkeypatch: pytest.MonkeyPatch):
    real = Path.read_text

    def fake(self: Path, *args, **kwargs):
        if self.as_posix() == "/proc/mounts":
            return MOUNTS
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake)


@posix_only
@pytest.mark.parametrize(
    ("path", "fstype", "remote"),
    [
        ("/nas/clips/v.mp4", "nfs4", True),
        ("/nas/local/v.mp4", "ext4", False),  # deepest mount wins
        ("/nasty/v.mp4", "ext4", False),  # "/nas" is not a prefix of "/nasty"
        ("/mnt/my share/v.mp4", "cifs", True),  # \040 is a space
        ("/run/user/1000/gvfs/smb-share/v.mp4", "fuse.gvfsd-fuse", True),
    ],
)
def test_mount_picks_the_deepest_mount(proc_mounts, path, fstype, remote):
    assert stage._mount(Path(path))[1] == fstype
    assert stage.is_remote(Path(path)) is remote


@posix_only
def test_mount_without_proc_mounts_is_local(monkeypatch):
    def boom(self, *args, **kwargs):
        raise FileNotFoundError("/proc/mounts")

    monkeypatch.setattr(Path, "read_text", boom)
    assert stage._mount(Path("/x/v.mp4")) == ("", "")
    assert stage.is_remote(Path("/x/v.mp4")) is False


@pytest.mark.parametrize(("drive", "remote"), [("C:", False), ("d:", False), ("F:", True)])
def test_is_remote_windows_uses_the_drive_letter(plat, drive, remote):
    plat("win32")
    assert stage.is_remote(PureWindowsPath(drive + "\\v.mp4")) is remote  # type: ignore[arg-type]


# -- stage: storage label ---------------------------------------------------------------------


@posix_only
@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("/nas/clips/v.mp4", "/nas (nfs4)"),
        ("/nas/local/v.mp4", "/nas/local (ext4)"),
        ("/mnt/my share/v.mp4", "/mnt/my share (cifs)"),
    ],
)
def test_storage_label_is_mount_point_and_fstype(proc_mounts, path, label):
    assert stage.storage_label(Path(path)) == label


@posix_only
def test_storage_label_of_an_unknown_mount_is_the_parent_folder(monkeypatch):
    def boom(self, *args, **kwargs):
        raise FileNotFoundError("/proc/mounts")

    monkeypatch.setattr(Path, "read_text", boom)
    assert stage.storage_label(Path("/x/y/v.mp4")) == str(Path("/x/y"))


def test_storage_label_windows_is_the_drive(plat):
    plat("win32")
    assert stage.storage_label(PureWindowsPath("F:\\Footage\\v.mp4")) == "F:"  # type: ignore[arg-type]


# -- cache: file mode of atomically written files ---------------------------------------------


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


@posix_only
def test_save_json_new_file_gets_the_mode_a_plain_write_would(tmp_path, umask_022):
    target = tmp_path / "new.json"
    cache.save_json(target, {"a": 1})
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


@posix_only
def test_save_json_keeps_the_mode_of_an_existing_file(tmp_path, umask_022):
    target = write(tmp_path / "old.json", "{}")
    target.chmod(0o640)
    cache.save_json(target, {"b": 2})
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 2}


@posix_only
def test_match_mode_follows_the_umask(tmp_path):
    old = os.umask(0o077)
    try:
        tmp = write(tmp_path / "t.tmp")
        cache.match_mode(tmp, tmp_path / "missing.json")
    finally:
        os.umask(old)
    assert stat.S_IMODE(tmp.stat().st_mode) == 0o600


# -- telemetry.extract: "not staged" warning -------------------------------------------------------


@pytest.fixture
def extract_env(tmp_path, monkeypatch, plat):
    plat("linux")
    monkeypatch.setattr(telemetry, "_compute", lambda *_a, **_k: None)
    src = write(tmp_path / "footage" / "ride.mp4")
    return src


def _not_staged(caplog) -> list[str]:
    return [m for m in messages(caplog) if "is not staged" in m]


def test_extract_local_source_does_not_warn_about_staging(extract_env, monkeypatch, caplog):
    monkeypatch.setattr(stage, "_mount", lambda _p: ("/", "ext4"))
    with caplog.at_level(logging.INFO, logger=telemetry.logger.name):
        assert telemetry.extract(extract_env, 10.0) is None
    assert _not_staged(caplog) == []


def test_extract_remote_source_warns_with_the_storage_label(extract_env, monkeypatch, caplog):
    monkeypatch.setattr(stage, "_mount", lambda _p: ("/mnt/nas", "nfs4"))
    with caplog.at_level(logging.INFO, logger=telemetry.logger.name):
        assert telemetry.extract(extract_env, 10.0) is None
    warned = _not_staged(caplog)
    assert len(warned) == 1
    assert "ride.mp4" in warned[0]
    assert "/mnt/nas (nfs4)" in warned[0]


# -- "no readable GoPro telemetry" is logged once -------------------------------------------------------


def test_no_readable_telemetry_is_logged_once_at_info(tmp_path, monkeypatch, caplog):
    class Parser:
        def __init__(self, _path):
            raise RuntimeError("no GoPro MET track")

    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=Parser))
    video = tmp_path / "plain.mp4"
    with caplog.at_level(logging.DEBUG):
        assert telemetry._compute(video, 10.0, telemetry.TelemetryParams()) is None
        assert crash._compute_imu(video, 10.0) is None
    hits = [r for r in caplog.records if "no readable GoPro telemetry" in r.message]
    assert [r.levelno for r in hits if r.levelno >= logging.INFO] == [logging.INFO]
    assert len(hits) == 2  # the second one is DEBUG, hidden at the default level
    caplog.clear()
    with caplog.at_level(logging.INFO):
        crash._compute_imu(video, 10.0)
    assert not [r for r in caplog.records if "no readable GoPro" in r.message]
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        crash._compute_imu(video, 10.0, force=True)
    assert [r.levelno for r in caplog.records if "no readable GoPro" in r.message] == [logging.INFO]


# -- ffmpeg_utils ---------------------------------------------------------------------------


def test_ffmpeg_not_found_on_linux_points_to_apt_and_env(plat, monkeypatch):
    plat("linux")
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _n: None)
    with pytest.raises(errors.FFmpegError) as exc:
        ffmpeg_utils.find_tool("ffmpeg")
    msg = str(exc.value)
    assert "apt" in msg
    assert "CUTPOINT_FFMPEG" in msg
    assert "winget" not in msg.lower()


# -- assemble.caption_font ----------------------------------------------------------------


def test_caption_font_uses_fc_match_family_on_linux(plat, monkeypatch):
    plat("linux")
    monkeypatch.setattr(assemble.shutil, "which", lambda _n: "/usr/bin/fc-match")
    calls = []

    def run(cmd, **_kw):
        calls.append(cmd)
        return SimpleNamespace(stdout="Noto Sans,Noto Sans Regular")

    monkeypatch.setattr(assemble.subprocess, "run", run)
    assert assemble.caption_font() == "Noto Sans"  # first family only: ',' splits an ASS style
    assert calls[0][-1] == "sans-serif"


@pytest.mark.parametrize("exe", [None, "/usr/bin/fc-match"], ids=["no-fc-match", "empty-output"])
def test_caption_font_without_a_usable_fc_match_falls_back_with_warning(
    plat, monkeypatch, caplog, exe
):
    plat("linux")
    monkeypatch.setattr(assemble.shutil, "which", lambda _n: exe)
    monkeypatch.setattr(assemble.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=""))
    with caplog.at_level(logging.WARNING, logger=assemble.logger.name):
        assert assemble.caption_font() == "DejaVu Sans"
    assert any(r.levelno == logging.WARNING and "fc-match" in r.message for r in caplog.records)


def test_write_ass_uses_the_caption_font(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "caption_font", lambda: "Test Sans")
    out = tmp_path / "c.ass"
    assemble.write_ass([(0.0, 1.0, "hi")], 1920, 1080, out)
    assert "Style: Default,Test Sans," in out.read_text(encoding="utf-8")


# -- pano/x3.imu_source -------------------------------------------------------------------


@pytest.mark.parametrize(("prefix", "ext"), [("VID", "insv"), ("vid", "INSV")])
def test_imu_source_finds_the_00_sibling_case_insensitively(tmp_path, ext, prefix):
    real = write(tmp_path / f"{prefix}_20240101_120000_00_001.{ext}")
    lens10 = tmp_path / "VID_20240101_120000_10_001.insv"
    got = x3.imu_source(lens10)
    assert got is not None
    # exact on-disk spelling (WindowsPath == ignores case), parent as passed
    assert str(got) == str(real)


def test_imu_source_matches_a_unique_00_by_number_only(tmp_path):
    write(tmp_path / "vid_20240101_120005_00_002.INSV")
    got = x3.imu_source(tmp_path / "LRV_20240101_120000_11_002.lrv")
    assert got is not None
    assert got.name == "vid_20240101_120005_00_002.INSV"


def test_imu_source_ambiguous_or_missing_is_none(tmp_path):
    write(tmp_path / "VID_20240101_120005_00_003.insv")
    write(tmp_path / "VID_20240101_120009_00_003.insv")
    assert x3.imu_source(tmp_path / "LRV_20240101_120000_11_003.lrv") is None
    assert x3.imu_source(tmp_path / "LRV_20240101_120000_11_004.lrv") is None
    assert x3.imu_source(tmp_path / "plain.mp4") is None


# -- pano/stitch.sdk_env ------------------------------------------------------------------


def test_sdk_env_adds_lib_dir_to_ld_library_path_on_linux(tmp_path, plat, monkeypatch):
    plat("linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old")
    lib = tmp_path / "MediaSDK" / "lib"
    write(lib / "libMediaSDK.so.3")
    exe = tmp_path / "MediaSDK" / "bin" / "MediaSDKTest"
    env = stitch.sdk_env(exe)
    assert env is not None
    assert env["LD_LIBRARY_PATH"] == f"{lib}{os.pathsep}/old"


def test_sdk_env_lib_next_to_the_exe_without_existing_path(tmp_path, plat, monkeypatch):
    plat("linux")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    write(tmp_path / "lib" / "libx.so")
    env = stitch.sdk_env(tmp_path / "MediaSDKTest")
    assert env is not None
    assert env["LD_LIBRARY_PATH"] == str(tmp_path / "lib")


def test_sdk_env_is_none_without_shared_libs_or_off_linux(tmp_path, plat):
    plat("linux")
    write(tmp_path / "lib" / "readme.txt")
    assert stitch.sdk_env(tmp_path / "MediaSDKTest") is None
    write(tmp_path / "lib" / "libx.so")
    plat("win32")
    assert stitch.sdk_env(tmp_path / "MediaSDKTest.exe") is None


@posix_only
def test_save_json_survives_a_file_system_that_rejects_chmod(tmp_path, monkeypatch):
    def nope(self, *args, **kwargs):
        raise OSError(errno.ENOTSUP, "chmod not supported")

    monkeypatch.setattr(Path, "chmod", nope)
    target = tmp_path / "fuse.json"
    cache.save_json(target, {"a": [1, 2]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": [1, 2]}


STATUS = "Name:\tpython\nUmask:\t0027\nState:\tR\n"


def _fake_status(monkeypatch, text):
    real = Path.read_text

    def fake(self: Path, *args, **kwargs):
        if self.as_posix() == "/proc/self/status":
            if text is None:
                raise FileNotFoundError("/proc/self/status")
            return text
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake)


@posix_only
def test_umask_is_read_from_proc_self_status(monkeypatch):
    _fake_status(monkeypatch, STATUS)
    assert cache._umask() == 0o027


@posix_only
@pytest.mark.parametrize("text", [None, "Name:\tpython\nState:\tR\n", "Umask:\tzz\n"])
def test_umask_falls_back_to_the_real_umask(monkeypatch, text, umask_022):
    _fake_status(monkeypatch, text)
    assert cache._umask() == 0o022
    assert os.umask(0o022) == 0o022  # the probe restored it


# -- procs.pdeathsig_hook --------------------------------------------------------------------------


def test_pdeathsig_hook_is_none_off_linux(plat):
    plat("win32")
    assert procs.pdeathsig_hook() is None
    plat("darwin")
    assert procs.pdeathsig_hook() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_pdeathsig_hook_is_callable_on_linux():
    assert callable(procs.pdeathsig_hook())  # not invoked: it would arm the test runner


# -- transcribe: Linux CUDA preload ----------------------------------------------------------------


@pytest.fixture
def fresh_cuda_setup(monkeypatch, plat):
    monkeypatch.setattr(transcribe, "_dll_dirs_added", False)
    plat("linux")


def test_cuda_preload_without_nvidia_wheels_does_nothing(fresh_cuda_setup, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(tmp_path)])

    def boom(*_a, **_k):
        raise AssertionError("nothing should be loaded")

    monkeypatch.setattr(transcribe.ctypes, "CDLL", boom)
    transcribe._add_cuda_dll_dirs()
    assert transcribe._dll_dirs_added is True


def test_cuda_preload_survives_libraries_that_fail_to_load(fresh_cuda_setup, tmp_path, monkeypatch):
    write(tmp_path / "nvidia" / "cublas" / "lib" / "libcublas.so.12")
    write(tmp_path / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    tried: list[str] = []

    def cdll(name, **_kw):
        tried.append(Path(name).name)
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(transcribe.ctypes, "CDLL", cdll)
    transcribe._add_cuda_dll_dirs()  # must neither raise nor retry forever
    assert sorted(tried) == ["libcublas.so.12", "libcudnn.so.9"]


# -- the sh wrapper ----------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "cutpoint"


@pytest.mark.slow
@posix_only
def test_wrapper_runs_the_venv_python(tmp_path):
    if not (ROOT / ".venv" / "bin" / "python").exists():
        pytest.skip("no .venv/bin/python in the repository (run uv sync)")
    done = subprocess.run(
        ["sh", str(WRAPPER), "--help"], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "cutpoint" in done.stdout.lower()


@pytest.mark.slow
@posix_only
def test_wrapper_without_a_venv_explains_uv_sync(tmp_path):
    copy = tmp_path / "cutpoint"
    shutil.copy(WRAPPER, copy)
    done = subprocess.run(
        ["sh", str(copy), "--help"], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 1
    assert "uv sync" in done.stderr
    assert ".venv/bin/python" in done.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="mapped drives are a Windows feature")
@pytest.mark.parametrize(
    ("key", "typed"),
    [
        pytest.param(
            r"\\nas\share\Footage",
            r"Q:\Footage\x.mp4",
            # on Windows, Path.resolve() of the unreachable \\nas\share does SMB lookups (~2.7 s)
            marks=pytest.mark.slow,
            id="unc-key-drive-input",
        ),
        pytest.param(r"Q:\Footage", r"\\nas\share\Footage\x.mp4", id="drive-key-unc-input"),
    ],
)
def test_remap_source_matches_a_disconnected_mapped_drive_in_either_form(
    home, tmp_path, monkeypatch, key, typed
):
    """Q: is mapped to a NAS share but not connected here, so it cannot be resolved."""
    monkeypatch.setattr(config, "_drive_remote", lambda d: r"\\nas\share" if d == "Q:" else None)
    copy = write(tmp_path / "copy" / "x.mp4")
    set_config(home, source_remap={key: str(tmp_path / "copy")})
    assert config.remap_source(typed) == copy


@pytest.mark.skipif(sys.platform != "win32", reason="mapped drives are a Windows feature")
def test_stage_key_of_a_copy_matches_the_unc_original_of_a_mapped_drive(
    home, tmp_path, monkeypatch
):
    """A staged copy of a Q: file is found by the UNC path find_videos returns."""
    monkeypatch.setattr(config, "_drive_remote", lambda d: r"\\nas\share" if d == "Q:" else None)
    copy = write(tmp_path / "copy" / "x.mp4")
    set_config(home, source_remap={"Q:/Footage": str(tmp_path / "copy")})
    assert stage.stage_key(copy) == stage.stage_key("//nas/share/Footage/x.mp4")


@posix_only
@pytest.mark.parametrize(
    ("typed", "rest"),
    [
        (r"D:\Videos\Trip\x.mp4", "Trip/x.mp4"),
        ("D:/Videos/Trip/x.mp4", "Trip/x.mp4"),
        (r"d:\videos\Trip\x.mp4", "Trip/x.mp4"),
        (r"D:\VideosX\x.mp4", None),
    ],
    ids=["backslashes", "forward-slashes", "other-case", "sibling-prefix-not-matched"],
)
def test_remap_source_matches_a_windows_key_on_posix(home, tmp_path, typed, rest):
    """#149: a ``D:\\Videos`` key was made absolute onto the cwd, so it never matched."""
    copy = write(tmp_path / "copy" / "Trip" / "x.mp4")
    set_config(home, source_remap={r"D:\Videos": str(tmp_path / "copy")})
    if rest is None:
        assert config.remap_source(typed) == Path(typed)
        return
    assert config.remap_source(typed) == tmp_path / "copy" / rest == copy
    assert config.canonical_source(copy) == r"D:\Videos\Trip\x.mp4"
