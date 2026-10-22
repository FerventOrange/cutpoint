"""Platform behaviour: path keys, Linux mounts, tool lookup, llama.cpp setup, the sh wrapper.

``sys.platform`` is monkeypatched wherever the code branches on it, so every test is
valid on both Linux and Windows CI. Tests that need real POSIX features are skipped on Windows.
"""

from __future__ import annotations

import errno
import io
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from conftest import write
from cutpoint import assemble, cache, config, crash, ffmpeg_utils, stage, telemetry, transcribe
from cutpoint.events import llama_server, setup
from cutpoint.events import paths as ev_paths
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


def test_norm_path_posix_keeps_case_and_slashes(plat):
    plat("linux")
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


def test_source_path_windows_spelling_is_a_windows_path(plat):
    plat("linux")
    assert isinstance(config.source_path("C:\\a\\b.mp4"), PureWindowsPath)
    assert not isinstance(config.source_path("/x/y.mp4"), PureWindowsPath)


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


def _remap(home: Path, monkeypatch: pytest.MonkeyPatch, old: str, new: str) -> None:
    write(home / "config.json", json.dumps({"source_remap": {old: new}}))
    config.get_config.cache_clear()
    # The prefix forms depend on the real cwd / file system; keep just the typed form.
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))


def test_canonical_source_windows_is_case_insensitive_with_backslashes(home, plat, monkeypatch):
    plat("win32")
    _remap(home, monkeypatch, "D:\\Footage", "E:\\Footage-copy")
    assert config.canonical_source("e:/FOOTAGE-copy/Sub/x.mp4") == "D:\\Footage\\Sub\\x.mp4"
    assert config.canonical_source("D:\\Other\\x.mp4") == "D:\\Other\\x.mp4"


def test_canonical_source_posix_is_case_sensitive(home, plat, monkeypatch):
    plat("linux")
    _remap(home, monkeypatch, "/data/Footage", "/mnt/copy")
    assert config.canonical_source("/mnt/copy/Sub/x.mp4") == "/data/Footage/Sub/x.mp4"
    assert config.canonical_source("/mnt/COPY/Sub/x.mp4") == "/mnt/COPY/Sub/x.mp4"
    assert config.canonical_source("/mnt/copy2/x.mp4") == "/mnt/copy2/x.mp4"  # not a prefix


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
        ("/home/me/v.mp4", "ext4", False),
        ("/nas/clips/v.mp4", "nfs4", True),
        ("/nas/local/v.mp4", "ext4", False),  # deepest mount wins
        ("/nas", "nfs4", True),
        ("/nasty/v.mp4", "ext4", False),  # "/nas" is not a prefix of "/nasty"
        ("/mnt/my share/v.mp4", "cifs", True),  # \040 is a space
        ("/media/disk/v.mp4", "ext4", False),
        ("/run/user/1000/gvfs/smb-share/v.mp4", "fuse.gvfsd-fuse", True),
        ("/mnt/cloud/v.mp4", "fuse.rclone", True),
    ],
)
def test_mount_fstype_picks_the_deepest_mount(proc_mounts, path, fstype, remote):
    assert stage.mount_fstype(Path(path)) == fstype
    assert crash._is_remote(Path(path)) is remote


@posix_only
def test_mount_fstype_without_proc_mounts_is_local(monkeypatch):
    def boom(self, *args, **kwargs):
        raise FileNotFoundError("/proc/mounts")

    monkeypatch.setattr(Path, "read_text", boom)
    assert stage.mount_fstype(Path("/x/v.mp4")) == ""
    assert crash._is_remote(Path("/x/v.mp4")) is False


@pytest.mark.parametrize(("drive", "remote"), [("C:", False), ("d:", False), ("F:", True)])
def test_is_remote_windows_uses_the_drive_letter(plat, drive, remote):
    plat("win32")
    assert crash._is_remote(PureWindowsPath(drive + "\\v.mp4")) is remote  # type: ignore[arg-type]


# -- stage: storage label ---------------------------------------------------------------------


def test_stage_helpers_are_the_ones_crash_uses():
    assert crash._is_remote is stage.is_remote


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


def _not_staged(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.levelno == logging.WARNING and "is not staged" in r.message
    ]


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
    assert "ride.mp4" in warned[0].message
    assert "/mnt/nas (nfs4)" in warned[0].message


# -- "no readable GoPro telemetry" is logged once -------------------------------------------------------


def test_no_readable_telemetry_is_logged_once_at_info(tmp_path, monkeypatch, caplog):
    class Parser:
        def __init__(self, _path):
            raise RuntimeError("no GoPro MET track")

    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=Parser))
    video = tmp_path / "plain.mp4"
    with caplog.at_level(logging.DEBUG):
        assert telemetry._compute(video, 10.0, telemetry.TelemetryParams()) is None
        assert crash._parse_imu(video) is None
    hits = [r for r in caplog.records if "no readable GoPro telemetry" in r.message]
    assert [r.levelno for r in hits if r.levelno >= logging.INFO] == [logging.INFO]
    assert len(hits) == 2  # the second one is DEBUG, hidden at the default level
    caplog.clear()
    with caplog.at_level(logging.INFO):
        crash._parse_imu(video)
    assert not [r for r in caplog.records if "no readable GoPro" in r.message]


# -- ffmpeg_utils ---------------------------------------------------------------------------


def test_ffmpeg_not_found_on_linux_points_to_apt_and_env(plat, monkeypatch):
    plat("linux")
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _n: None)
    with pytest.raises(ffmpeg_utils.FFmpegError) as exc:
        ffmpeg_utils.find_tool("ffmpeg")
    msg = str(exc.value)
    assert "apt" in msg
    assert "CUTPOINT_FFMPEG" in msg
    assert "winget" not in msg.lower()


# -- assemble.caption_font ----------------------------------------------------------------


def test_caption_font_is_arial_on_windows(plat):
    plat("win32")
    assert assemble.caption_font() == "Arial"


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


def test_caption_font_without_fc_match_falls_back_with_warning(plat, monkeypatch, caplog):
    plat("linux")
    monkeypatch.setattr(assemble.shutil, "which", lambda _n: None)
    with caplog.at_level(logging.WARNING, logger=assemble.logger.name):
        assert assemble.caption_font() == "DejaVu Sans"
    assert any(r.levelno == logging.WARNING and "fc-match" in r.message for r in caplog.records)


def test_caption_font_empty_fc_match_output_falls_back(plat, monkeypatch, caplog):
    plat("linux")
    monkeypatch.setattr(assemble.shutil, "which", lambda _n: "/usr/bin/fc-match")
    monkeypatch.setattr(assemble.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=""))
    with caplog.at_level(logging.WARNING, logger=assemble.logger.name):
        assert assemble.caption_font() == "DejaVu Sans"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_write_ass_uses_the_caption_font(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "caption_font", lambda: "Test Sans")
    out = tmp_path / "c.ass"
    assemble.write_ass([(0.0, 1.0, "hi")], 1920, 1080, out)
    assert "Style: Default,Test Sans," in out.read_text(encoding="utf-8")


# -- pano/x3.imu_source -------------------------------------------------------------------


@pytest.mark.parametrize("ext", ["insv", "INSV"])
@pytest.mark.parametrize("prefix", ["VID", "vid"])
def test_imu_source_finds_the_00_sibling_case_insensitively(tmp_path, ext, prefix):
    real = write(tmp_path / f"{prefix}_20240101_120000_00_001.{ext}")
    lens10 = tmp_path / "VID_20240101_120000_10_001.insv"
    got = x3.imu_source(lens10)
    assert got is not None
    assert got.samefile(real)
    if sys.platform.startswith("linux"):
        # Case-sensitive FS: the on-disk spelling is returned. Elsewhere the built name already exists.
        assert got.name == real.name


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


# -- events/setup: llama.cpp assets ---------------------------------------------------------

TAG = "b9000"
PRE = f"llama-{TAG}-bin-ubuntu-"


def _release(*names: str) -> dict[str, tuple[str, int]]:
    return {n: (f"https://example.invalid/{n}", 1) for n in names}


FULL = _release(
    f"{PRE}x64.tar.gz",
    f"{PRE}vulkan-x64.tar.gz",
    f"{PRE}cuda-12.8-x64.tar.gz",
    f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    f"{PRE}cuda-13.0-x64.tar.gz",
    f"cudart-llama-{TAG}-bin-ubuntu-cuda-13.0-x64.tar.gz",
    f"{PRE}arm64.tar.gz",
)


def test_pick_linux_assets_nvidia_takes_the_oldest_cuda_with_cudart():
    flavor, picked = setup._pick_linux_assets(TAG, FULL, "x64", nvidia=True)
    assert flavor == "linux-cuda-12.8"
    assert picked["bin"][0].endswith(f"{PRE}cuda-12.8-x64.tar.gz")
    assert picked["rt"][0].endswith(f"cudart-llama-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz")


def test_pick_linux_assets_skips_a_cuda_build_without_cudart():
    assets = {k: v for k, v in FULL.items() if "cudart" not in k or "13.0" in k}
    flavor, picked = setup._pick_linux_assets(TAG, assets, "x64", nvidia=True)
    assert flavor == "linux-cuda-13.0"
    assert set(picked) == {"bin", "rt"}


def test_pick_linux_assets_without_nvidia_prefers_vulkan_then_cpu():
    flavor, picked = setup._pick_linux_assets(TAG, FULL, "x64", nvidia=False)
    assert (flavor, set(picked)) == ("linux-vulkan", {"bin"})
    no_vulkan = {k: v for k, v in FULL.items() if "vulkan" not in k}
    flavor, picked = setup._pick_linux_assets(TAG, no_vulkan, "x64", nvidia=False)
    assert (flavor, set(picked)) == ("linux-cpu", {"bin"})


def test_pick_linux_assets_arch_and_zip_fallback_and_none():
    flavor, _ = setup._pick_linux_assets(TAG, FULL, "arm64", nvidia=True)
    assert flavor == "linux-cpu"  # no CUDA / Vulkan build for arm64 in this release
    zips = _release(f"{PRE}vulkan-x64.zip")
    assert setup._pick_linux_assets(TAG, zips, "x64", nvidia=False)[0] == "linux-vulkan"
    assert (
        setup._pick_linux_assets(TAG, _release("llama-b9000-bin-win-cpu-x64.zip"), "x64", True)
        is None
    )


@pytest.mark.parametrize(
    ("stamp", "plat_name", "fits"),
    [
        ("b1 cuda-12.4\n", "win", True),
        ("b1 cuda-12.4", "linux", False),
        ("b1 linux-vulkan", "linux", True),
        ("b1 linux-cuda-12.8", "win", False),
    ],
)
def test_stamp_fits_only_the_platform_it_was_installed_for(stamp, plat_name, fits):
    assert setup._stamp_fits(stamp, plat_name) is fits


def test_llama_platform_rejects_other_platforms(plat):
    plat("darwin")
    if os.name == "nt":
        pytest.skip("os.name is 'nt' here")
    with pytest.raises(RuntimeError, match="CUTPOINT_LLAMA_SERVER"):
        setup._llama_platform()


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


# -- events/setup._extract --------------------------------------------------------------------


def _tar(path: Path, entries: dict[str, tuple[bytes | None, int]]) -> Path:
    """``entries``: name -> (content, mode); content None makes a directory."""
    with tarfile.open(path, "w:gz") as tf:
        for name, (data, mode) in entries.items():
            info = tarfile.TarInfo(name)
            info.mode = mode
            if data is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return path


def test_extract_tarball_strips_the_top_level_folder(tmp_path):
    arc = _tar(
        tmp_path / "llama.tar.gz",
        {
            f"llama-{TAG}": (None, 0o755),
            f"llama-{TAG}/llama-server": (b"#!/bin/sh\n", 0o755),
            f"llama-{TAG}/libggml.so": (b"lib", 0o644),
        },
    )
    dest = tmp_path / "out"
    setup._extract(arc, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["libggml.so", "llama-server"]
    if sys.platform != "win32":
        assert (dest / "llama-server").stat().st_mode & stat.S_IXUSR


def test_extract_tarball_with_several_top_level_entries_is_kept_as_is(tmp_path):
    arc = _tar(tmp_path / "a.tar.gz", {"one/a.txt": (b"a", 0o644), "two.txt": (b"b", 0o644)})
    dest = tmp_path / "out"
    setup._extract(arc, dest)
    assert (dest / "one" / "a.txt").is_file()
    assert (dest / "two.txt").is_file()


@posix_only
def test_extract_zip_restores_the_exec_bit(tmp_path):
    arc = tmp_path / "llama.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        exe = zipfile.ZipInfo("llama-server")
        exe.external_attr = (stat.S_IFREG | 0o755) << 16
        zf.writestr(exe, "#!/bin/sh\n")
        plain = zipfile.ZipInfo("notes.txt")
        plain.external_attr = (stat.S_IFREG | 0o644) << 16
        zf.writestr(plain, "x")
    dest = tmp_path / "out"
    setup._extract(arc, dest)
    assert (dest / "llama-server").stat().st_mode & 0o777 == 0o755
    assert (dest / "notes.txt").stat().st_mode & stat.S_IXUSR == 0


def test_extract_zip_refuses_path_traversal(tmp_path):
    arc = tmp_path / "evil.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe path"):
        setup._extract(arc, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_zip_refuses_a_sibling_directory_sharing_the_prefix(tmp_path):
    arc = tmp_path / "evil.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        zf.writestr("../llama.cpp-x/f", "x")
    with pytest.raises(RuntimeError, match="unsafe path"):
        setup._extract(arc, tmp_path / "llama.cpp")
    assert not (tmp_path / "llama.cpp-x").exists()


@posix_only
def test_extract_zip_caps_modes_at_755(tmp_path):
    arc = tmp_path / "m.zip"
    with zipfile.ZipFile(arc, "w") as zf:
        wide = zipfile.ZipInfo("wide.bin")
        wide.external_attr = (stat.S_IFREG | 0o777) << 16
        zf.writestr(wide, "x")
        folder = zipfile.ZipInfo("sub/")
        folder.external_attr = (stat.S_IFDIR | 0o777) << 16
        zf.writestr(folder, "")
        zf.writestr("sub/f.txt", "y")
    dest = tmp_path / "out"
    setup._extract(arc, dest)
    assert stat.S_IMODE((dest / "wide.bin").stat().st_mode) == 0o755
    assert (dest / "sub").is_dir()
    assert (dest / "sub" / "f.txt").read_text() == "y"


@posix_only
def test_install_llama_only_chmods_its_own_binary(tmp_path, monkeypatch):
    other = write(tmp_path / "other" / "llama-server")
    other.chmod(0o644)
    own = write(ev_paths.llama_dir() / "build" / "llama-server")
    own.chmod(0o644)
    monkeypatch.setenv("CUTPOINT_LLAMA_SERVER", str(other))
    monkeypatch.setattr(setup, "_llama_platform", lambda: "linux")
    monkeypatch.setattr(setup, "_latest_llama_assets", lambda _p: ("b1", "cpu", {}))
    setup.install_llama(force=True)
    assert stat.S_IMODE(other.stat().st_mode) == 0o644
    assert stat.S_IMODE(own.stat().st_mode) & 0o111 == 0o111
    assert (ev_paths.llama_dir() / "VERSION.txt").read_text().strip() == "b1 cpu"


# -- events/paths.llama_server_exe ---------------------------------------------------------------

EXE_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"


def test_llama_server_exe_env_override_wins(tmp_path, monkeypatch):
    bundled = write(ev_paths.llama_dir() / EXE_NAME)
    custom = write(tmp_path / "mine" / "llama-server")
    monkeypatch.setenv("CUTPOINT_LLAMA_SERVER", str(custom))
    monkeypatch.setattr(ev_paths.shutil, "which", lambda _n: str(tmp_path / "onpath"))
    assert ev_paths.llama_server_exe() == custom
    assert bundled.exists()


def test_llama_server_exe_bundled_beats_path(tmp_path, monkeypatch):
    bundled = write(ev_paths.llama_dir() / "build" / EXE_NAME)
    monkeypatch.setattr(ev_paths.shutil, "which", lambda _n: str(tmp_path / "onpath"))
    assert ev_paths.llama_server_exe() == bundled


def test_llama_server_exe_falls_back_to_path_then_the_expected_location(tmp_path, monkeypatch):
    onpath = write(tmp_path / "bin" / "llama-server")
    monkeypatch.setattr(ev_paths.shutil, "which", lambda _n: str(onpath))
    assert ev_paths.llama_server_exe() == onpath
    monkeypatch.setattr(ev_paths.shutil, "which", lambda _n: None)
    assert ev_paths.llama_server_exe() == ev_paths.llama_dir() / EXE_NAME


# -- events/llama_server._pdeathsig_hook -----------------------------------------------------------


def test_pdeathsig_hook_is_none_off_linux(plat):
    plat("win32")
    assert llama_server._pdeathsig_hook() is None
    plat("darwin")
    assert llama_server._pdeathsig_hook() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_pdeathsig_hook_is_callable_on_linux():
    assert callable(llama_server._pdeathsig_hook())  # not invoked: it would arm the test runner


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


@posix_only
def test_wrapper_runs_the_venv_python(tmp_path):
    if not (ROOT / ".venv" / "bin" / "python").exists():
        pytest.skip("no .venv/bin/python in the repository (run uv sync)")
    done = subprocess.run(
        ["sh", str(WRAPPER), "--help"], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "cutpoint" in done.stdout.lower()


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
