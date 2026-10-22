"""events/setup: sha256-verified downloads and the staged llama.cpp install."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import pytest

from conftest import write
from cutpoint.events import paths as ev_paths
from cutpoint.events import setup

DATA = b"payload-bytes" * 50
DIGEST = hashlib.sha256(DATA).hexdigest()


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.as_uri()


# -- download(..., sha256=) ---------------------------------------------------------------------


@pytest.mark.parametrize("fix", [str.lower, str.upper])
def test_download_accepts_matching_hash_in_any_case(tmp_path, fix):
    url = write_bytes(tmp_path / "srv" / "a.bin", DATA)
    dst = setup.download(url, tmp_path / "out" / "a.bin", len(DATA), fix(DIGEST))
    assert dst.read_bytes() == DATA


def test_download_without_sha256_does_not_verify(tmp_path):
    url = write_bytes(tmp_path / "srv" / "a.bin", DATA)
    dst = setup.download(url, tmp_path / "out" / "a.bin")
    assert dst.read_bytes() == DATA


def test_download_without_sha256_keeps_existing_file_untouched(tmp_path, monkeypatch):
    dst = write(tmp_path / "out" / "a.bin", "anything")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert setup.download("file:///nope", dst, dst.stat().st_size) == dst
    assert dst.read_text() == "anything"


def test_download_existing_file_with_right_hash_is_not_fetched(tmp_path, monkeypatch):
    dst = tmp_path / "out" / "a.bin"
    dst.parent.mkdir()
    dst.write_bytes(DATA)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert setup.download("file:///nope", dst, len(DATA), DIGEST) == dst


def test_download_existing_file_with_wrong_hash_is_fetched_again(tmp_path, monkeypatch):
    url = write_bytes(tmp_path / "srv" / "a.bin", DATA)
    dst = tmp_path / "out" / "a.bin"
    dst.parent.mkdir()
    dst.write_bytes(b"X" * len(DATA))  # same size, corrupt content
    calls = []
    real = urllib.request.urlopen
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append(1) or real(*a, **k))
    setup.download(url, dst, len(DATA), DIGEST)
    assert len(calls) == 1
    assert dst.read_bytes() == DATA


def test_download_fresh_mismatch_raises_and_leaves_nothing(tmp_path):
    url = write_bytes(tmp_path / "srv" / "a.bin", b"tampered")
    dst = tmp_path / "out" / "a.bin"
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        setup.download(url, dst, None, DIGEST)
    assert not dst.exists()
    assert not dst.with_name("a.bin.part").exists()


def test_download_wrong_hash_after_redownload_still_raises(tmp_path):
    url = write_bytes(tmp_path / "srv" / "a.bin", b"tampered")
    dst = tmp_path / "out" / "a.bin"
    dst.parent.mkdir()
    dst.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        setup.download(url, dst, None, DIGEST)
    assert not dst.exists()


# -- _sha256_of -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ({"digest": "sha256:abc123"}, "abc123"),
        ({}, None),
        ({"digest": None}, None),
        ({"digest": ""}, None),
        ({"digest": "md5:abc123"}, None),
    ],
)
def test_sha256_of_asset(asset, expected):
    assert setup._sha256_of(asset) == expected


# -- install_llama: staging and swap --------------------------------------------------------------


def _fake_linux(monkeypatch, files: dict[str, str] | None = None, tag="b2"):
    """Linux install whose download/extract are stubbed; extract writes ``files``."""
    files = files or {"build/llama-server": "new"}
    assets = {"bin": ("https://example.invalid/b.tar.gz", 1, None)}
    monkeypatch.setattr(setup, "_llama_platform", lambda: "linux")
    monkeypatch.setattr(setup, "_latest_llama_assets", lambda _p: (tag, "linux-cpu-x64", assets))
    monkeypatch.setattr(setup, "download", lambda _u, dst, *_a: dst)

    def extract(_archive, dest):
        for name, text in files.items():
            write(dest / name, text)

    monkeypatch.setattr(setup, "_extract", extract)


def _dirs():
    llama = ev_paths.llama_dir()
    return llama, llama.with_name(llama.name + ".new"), llama.with_name(llama.name + ".old")


def test_install_llama_replaces_old_build_entirely(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "stale.so", "old")
    _fake_linux(monkeypatch)
    setup.install_llama(force=True)
    assert not (llama / "stale.so").exists()
    assert (llama / "build" / "llama-server").read_text() == "new"
    assert (llama / "VERSION.txt").read_text() == "b2 linux-cpu-x64\n"
    assert not staging.exists()
    assert not old.exists()


def test_install_llama_cleans_leftovers_of_earlier_runs(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(staging / "junk", "x")
    write(old / "VERSION.txt", "b0 linux-cpu-x64\n")
    write(old / "junk", "x")
    _fake_linux(monkeypatch)
    setup.install_llama(force=True)
    assert not staging.exists()
    assert not old.exists()
    assert not (llama / "junk").exists()
    assert (llama / "build" / "llama-server").exists()


def test_install_llama_restores_old_after_crash_between_renames(monkeypatch):
    llama, staging, old = _dirs()
    write(old / "VERSION.txt", "b1 linux-cpu-x64\n")  # llama.cpp itself is missing
    write(old / "keep.so", "old")
    _fake_linux(monkeypatch)
    seen = {}

    def extract(_a, dest):
        seen["restored"] = (llama / "keep.so").exists()  # restored before the new build starts
        write(dest / "build" / "llama-server", "new")
        raise OSError("disk full")

    monkeypatch.setattr(setup, "_extract", extract)
    with pytest.raises(OSError, match="disk full"):
        setup.install_llama(force=True)
    assert seen["restored"]
    assert (llama / "keep.so").read_text() == "old"  # the old install survives the failure
    assert not old.exists()
    assert not staging.exists()


def test_install_llama_failed_download_keeps_old_install_and_removes_staging(monkeypatch):
    llama, staging, _old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "keep.so", "old")
    _fake_linux(monkeypatch)

    def boom(*_a):
        raise RuntimeError("sha256 mismatch")

    monkeypatch.setattr(setup, "download", boom)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        setup.install_llama(force=True)
    assert (llama / "keep.so").read_text() == "old"
    assert (llama / "VERSION.txt").read_text() == "b1 linux-cpu-x64\n"
    assert not staging.exists()


def test_install_llama_rename_failure_restores_old_and_hints(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "keep.so", "old")
    _fake_linux(monkeypatch)
    real_rename = Path.rename

    def rename(self, target):
        if self == staging:  # the old install was already moved aside by then
            raise PermissionError("in use")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(RuntimeError, match="Stop cutpoint") as exc:
        setup.install_llama(force=True)
    assert isinstance(exc.value.__cause__, PermissionError)
    assert (llama / "keep.so").read_text() == "old"
    assert (llama / "VERSION.txt").read_text() == "b1 linux-cpu-x64\n"
    assert not staging.exists()
    assert not old.exists()


def test_install_llama_old_to_aside_rename_failure_keeps_old_install(monkeypatch):
    llama, staging, _old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "keep.so", "old")
    _fake_linux(monkeypatch)
    real_rename = Path.rename

    def rename(self, target):
        if self == llama:
            raise PermissionError("llama-server.exe is running")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(RuntimeError, match="Stop cutpoint"):
        setup.install_llama(force=True)
    assert (llama / "keep.so").read_text() == "old"
    assert not staging.exists()


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record instead of sleeping in _rename_retry."""
    rec: list[float] = []
    monkeypatch.setattr(setup.time, "sleep", rec.append)
    return rec


def test_install_llama_retries_transient_rename_lock(monkeypatch, sleeps):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "stale.so", "old")
    _fake_linux(monkeypatch)
    real_rename = Path.rename
    fails = {"left": 2}

    def rename(self, target):
        if self == staging and fails["left"]:
            fails["left"] -= 1
            raise PermissionError("antivirus scan")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    setup.install_llama(force=True)
    assert len(sleeps) == 2
    assert (llama / "build" / "llama-server").read_text() == "new"
    assert not (llama / "stale.so").exists()
    assert not staging.exists()
    assert not old.exists()


def test_rename_retry_gives_up_after_tries(tmp_path, sleeps):
    class Locked:
        calls = 0

        def rename(self, _dst):
            Locked.calls += 1
            raise PermissionError("locked")

    with pytest.raises(PermissionError):
        setup._rename_retry(Locked(), tmp_path / "x", tries=3)
    assert Locked.calls == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("files", [["foo.txt"], ["llama-server.log", "foo.txt"]])
def test_ours_rejects_dirs_without_stamp(tmp_path, files):
    for name in files:
        write(tmp_path / "d" / name, "x")
    assert not setup._ours(tmp_path / "d")


@pytest.mark.parametrize("files", [[], ["llama-server.log"], ["VERSION.txt", "foo.txt"]])
def test_ours_accepts_stamped_empty_or_log_only_dirs(tmp_path, files):
    (tmp_path / "d").mkdir()
    for name in files:
        write(tmp_path / "d" / name, "x")
    assert setup._ours(tmp_path / "d")


def test_ours_accepts_missing_dir(tmp_path):
    assert setup._ours(tmp_path / "nope")


def _lock(monkeypatch, name: str):
    """Make deleting any file called ``name`` fail, like an AV/process lock on Windows."""
    real_unlink = Path.unlink

    def unlink(self, *a, **k):
        if self.name == name:
            raise PermissionError("locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)


def test_rmtree_ours_removes_everything_including_stamp(tmp_path):
    d = tmp_path / "d"
    write(d / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(d / "a.so", "x")
    write(d / "sub" / "b.so", "x")
    setup._rmtree_ours(d)
    assert not d.exists()


def test_rmtree_ours_keeps_stamp_and_dir_when_a_file_is_locked(tmp_path, monkeypatch):
    d = tmp_path / "d"
    write(d / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(d / "locked.dll", "x")
    write(d / "free.so", "x")
    _lock(monkeypatch, "locked.dll")
    setup._rmtree_ours(d)  # best effort: must not raise
    assert sorted(p.name for p in d.iterdir()) == ["VERSION.txt", "locked.dll"]
    assert setup._ours(d)


def test_install_llama_survives_partial_cleanup_of_old_after_swap(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "locked.dll", "old")
    write(llama / "stale.so", "old")
    _fake_linux(monkeypatch)
    _lock(monkeypatch, "locked.dll")
    setup.install_llama(force=True)  # leftover .old is best effort
    assert (llama / "build" / "llama-server").read_text() == "new"
    assert (llama / "VERSION.txt").read_text() == "b2 linux-cpu-x64\n"
    assert not (llama / "locked.dll").exists()
    assert not staging.exists()
    assert (old / "VERSION.txt").exists()  # still ours, so the next run can clean it
    assert setup._ours(old)
    assert not (old / "stale.so").exists()


def test_install_llama_stuck_stamped_old_raises_before_downloading(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "keep.so", "current")
    write(old / "VERSION.txt", "b0 linux-cpu-x64\n")
    write(old / "locked.dll", "x")
    _fake_linux(monkeypatch)
    monkeypatch.setattr(setup, "download", lambda *a: pytest.fail("downloaded"))
    _lock(monkeypatch, "locked.dll")
    with pytest.raises(RuntimeError, match="Could not remove") as exc:
        setup.install_llama(force=True)
    assert str(old) in str(exc.value)
    assert (llama / "keep.so").read_text() == "current"
    assert (llama / "VERSION.txt").read_text() == "b1 linux-cpu-x64\n"
    assert (old / "VERSION.txt").exists()
    assert not staging.exists()


@pytest.mark.parametrize("which", ["llama", "old"])
def test_install_llama_refuses_unstamped_dir_and_keeps_it(monkeypatch, which):
    llama, staging, old = _dirs()
    target = llama if which == "llama" else old
    write(target / "mine.gguf", "precious")
    write(target / "notes.txt", "keep")
    _fake_linux(monkeypatch)
    monkeypatch.setattr(setup, "download", lambda *a: pytest.fail("downloaded"))
    with pytest.raises(RuntimeError, match="was not installed by cutpoint setup"):
        setup.install_llama(force=True)
    assert (target / "mine.gguf").read_text() == "precious"
    assert (target / "notes.txt").read_text() == "keep"
    assert not staging.exists()


def test_install_llama_replaces_dir_holding_only_server_log(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "llama-server.log", "log")
    _fake_linux(monkeypatch)
    setup.install_llama(force=True)
    assert not (llama / "llama-server.log").exists()
    assert (llama / "build" / "llama-server").read_text() == "new"
    assert not staging.exists()
    assert not old.exists()


def test_install_llama_already_installed_changes_nothing(monkeypatch):
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 linux-cpu-x64\n")
    write(llama / "keep.so", "old")
    _fake_linux(monkeypatch)
    monkeypatch.setattr(setup, "download", lambda *a: pytest.fail("downloaded"))
    monkeypatch.setattr(setup, "_extract", lambda *a: pytest.fail("extracted"))
    setup.install_llama()
    assert sorted(p.name for p in llama.iterdir()) == ["VERSION.txt", "keep.so"]
    assert not staging.exists()
    assert not old.exists()
    assert not (ev_paths.tools_dir() / "_downloads").exists()


def test_install_llama_passes_published_digest_to_download(monkeypatch):
    _fake_linux(monkeypatch)
    assets = {"bin": ("https://example.invalid/b.tar.gz", 7, "ab" * 32)}
    monkeypatch.setattr(setup, "_latest_llama_assets", lambda _p: ("b2", "linux-cpu-x64", assets))
    got = []
    monkeypatch.setattr(setup, "download", lambda *a: got.append(a) or a[1])
    setup.install_llama(force=True)
    assert got[0][2:] == (7, "ab" * 32)


# -- Windows flow end to end (zips served from file://, release list stubbed) ---------------------


def _zip(path: Path, files: dict[str, str]) -> tuple[str, int, str]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    data = buf.getvalue()
    return write_bytes(path, data), len(data), hashlib.sha256(data).hexdigest()


def _windows_release(tmp_path: Path, cuda_digest: str | None = None) -> list[dict]:
    tag = "b9100"
    assets = []
    for name, files in (
        (f"llama-{tag}-bin-win-cuda-12.4-x64.zip", {"llama-server.exe": "exe"}),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", {"cudart64_12.dll": "dll"}),
        (f"llama-{tag}-bin-win-cpu-x64.zip", {"llama-server.exe": "cpu"}),
    ):
        url, size, digest = _zip(tmp_path / "srv" / name, files)
        assets.append(
            {
                "name": name,
                "browser_download_url": url,
                "size": size,
                "digest": f"sha256:{cuda_digest or digest}",
            }
        )
    return [{"tag_name": tag, "assets": assets}]


def _serve_releases(monkeypatch, releases):
    real = urllib.request.urlopen

    def fake(req, *a, **k):
        if req == setup.GH_RELEASES:
            return io.BytesIO(json.dumps(releases).encode())
        return real(req, *a, **k)

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_install_llama_windows_picks_cuda_12_4_with_cudart(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(setup, "_llama_platform", lambda: "win")
    _serve_releases(monkeypatch, _windows_release(tmp_path))
    llama = ev_paths.llama_dir()
    write(llama / "VERSION.txt", "b1 cuda-12.4\n")
    write(llama / "stale.dll", "old")
    with caplog.at_level(logging.INFO):
        setup.install_llama(force=True)
    assert (llama / "llama-server.exe").read_text() == "exe"  # CUDA build, not cpu
    assert (llama / "cudart64_12.dll").exists()
    assert not (llama / "stale.dll").exists()
    assert (llama / "VERSION.txt").read_text() == "b9100 cuda-12.4\n"
    assert not (ev_paths.tools_dir() / "_downloads").exists()


def test_install_llama_windows_bad_digest_aborts_and_keeps_old(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "_llama_platform", lambda: "win")
    _serve_releases(monkeypatch, _windows_release(tmp_path, cuda_digest="0" * 64))
    llama = ev_paths.llama_dir()
    write(llama / "VERSION.txt", "b1 cuda-12.4\n")
    write(llama / "keep.dll", "old")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        setup.install_llama(force=True)
    assert (llama / "keep.dll").read_text() == "old"
    assert not llama.with_name(llama.name + ".new").exists()
