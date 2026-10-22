"""events/setup: sha256-verified llama.cpp downloads and the staged install (issues #35, #36).

Releases are served from file:// zips with only the GitHub release list stubbed.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pytest

from conftest import write
from cutpoint.events import paths as ev_paths
from cutpoint.events import setup

OLD_STAMP = "b1 cuda-12.4\n"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record instead of sleeping while renames are retried."""
    rec: list[float] = []
    monkeypatch.setattr(setup.time, "sleep", rec.append)
    return rec


def write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.as_uri()


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
        (f"llama-{tag}-bin-win-vulkan-x64.zip", {"llama-server.exe": "vulkan"}),
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


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Serve a Windows release; ``release(cuda_digest=...)`` can publish a wrong digest.

    ``nvidia=False`` makes setup see no usable NVIDIA driver.
    """

    def serve(cuda_digest: str | None = None, nvidia: bool = True) -> None:
        monkeypatch.setattr(setup, "nvidia_usable", lambda: nvidia)
        releases = _windows_release(tmp_path, cuda_digest)
        real = urllib.request.urlopen

        def fake(req, *a, **k):
            if req == setup.GH_RELEASES:
                return io.BytesIO(json.dumps(releases).encode())
            return real(req, *a, **k)

        monkeypatch.setattr(setup, "_llama_platform", lambda: "win")
        monkeypatch.setattr(urllib.request, "urlopen", fake)

    return serve


def _dirs():
    llama = ev_paths.llama_dir()
    return llama, llama.with_name(llama.name + ".new"), llama.with_name(llama.name + ".old")


def _old_install():
    llama = ev_paths.llama_dir()
    write(llama / "VERSION.txt", OLD_STAMP)
    write(llama / "keep.dll", "old")
    return llama


def _assert_old_kept(llama, staging, old):
    assert (llama / "keep.dll").read_text() == "old"
    assert (llama / "VERSION.txt").read_text() == OLD_STAMP
    assert not staging.exists()
    assert not old.exists()


# -- #35: downloads are verified ------------------------------------------------------------------


def test_corrupt_cached_archive_is_downloaded_again(tmp_path):
    data = b"payload-bytes" * 50
    url = write_bytes(tmp_path / "srv" / "a.bin", data)
    dst = tmp_path / "out" / "a.bin"
    dst.parent.mkdir()
    dst.write_bytes(b"X" * len(data))  # same size, corrupt content
    setup.download(url, dst, len(data), hashlib.sha256(data).hexdigest())
    assert dst.read_bytes() == data


def test_install_llama_bad_digest_aborts_and_leaves_nothing(release):
    release(cuda_digest="0" * 64)
    llama, staging, old = _dirs()
    _old_install()
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        setup.install_llama(force=True)
    _assert_old_kept(llama, staging, old)
    downloads = ev_paths.tools_dir() / "_downloads"
    assert not [p for p in downloads.rglob("*") if p.is_file()]  # no archive, no .part


# -- #36: staged install --------------------------------------------------------------------------


def test_install_llama_windows_picks_cuda_12_4_and_replaces_old_build(release):
    release()
    llama, staging, old = _dirs()
    write(llama / "VERSION.txt", "b1 cpu\n")
    write(llama / "stale.dll", "old")
    setup.install_llama(force=True)
    assert (llama / "llama-server.exe").read_text() == "exe"  # CUDA build, not cpu
    assert (llama / "cudart64_12.dll").exists()
    assert not (llama / "stale.dll").exists()
    assert (llama / "VERSION.txt").read_text() == "b9100 cuda-12.4\n"
    assert not staging.exists()
    assert not old.exists()


@pytest.mark.parametrize(
    ("nvidia", "override", "server", "stamp"),
    [
        (False, None, "vulkan", "b9100 win-vulkan\n"),  # AMD / Intel / no usable NVIDIA
        (True, "vulkan", "vulkan", "b9100 win-vulkan\n"),  # explicit override beats vendor
        (False, "cpu", "cpu", "b9100 win-cpu\n"),
    ],
)
def test_install_llama_windows_without_usable_nvidia_or_forced_uses_vulkan_or_cpu(
    release, monkeypatch, nvidia, override, server, stamp
):
    release(nvidia=nvidia)
    if override:
        monkeypatch.setenv("CUTPOINT_LLAMA_BACKEND", override)
    llama, _, _ = _dirs()
    setup.install_llama(force=True)
    assert (llama / "llama-server.exe").read_text() == server
    assert not (llama / "cudart64_12.dll").exists()
    assert (llama / "VERSION.txt").read_text() == stamp


def test_install_llama_forced_backend_replaces_a_different_installed_build(release, monkeypatch):
    release()
    llama = _old_install()  # stamp: b1 cuda-12.4
    monkeypatch.setenv("CUTPOINT_LLAMA_BACKEND", "vulkan")
    setup.install_llama()  # not force=True: the backend differs, so it is replaced anyway
    assert (llama / "VERSION.txt").read_text() == "b9100 win-vulkan\n"


def test_install_llama_auto_replaces_a_legacy_cuda_install_when_no_nvidia_is_usable(release):
    release(nvidia=False)
    llama = _old_install()  # stamp b1 cuda-12.4: what every Windows install was before Vulkan
    _, staging, old = _dirs()
    setup.install_llama()  # no force: the installed backend is not the one auto would pick now
    assert (llama / "VERSION.txt").read_text() == "b9100 win-vulkan\n"
    assert (llama / "llama-server.exe").read_text() == "vulkan"
    assert not (llama / "keep.dll").exists()
    assert not staging.exists() and not old.exists()


@pytest.mark.parametrize("locked", ["llama", "staging"])
def test_install_llama_locked_rename_keeps_previous_install(release, monkeypatch, locked):
    release()
    llama, staging, old = _dirs()
    _old_install()
    blocked = llama if locked == "llama" else staging
    real_rename = Path.rename

    def rename(self, target):
        if self == blocked:  # e.g. llama-server.exe still running
            raise PermissionError("in use")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(RuntimeError, match="Stop cutpoint") as exc:
        setup.install_llama(force=True)
    assert isinstance(exc.value.__cause__, PermissionError)
    _assert_old_kept(llama, staging, old)


def test_install_llama_retries_transient_rename_lock(release, monkeypatch, sleeps):
    release()
    llama, staging, _old = _dirs()
    _old_install()
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
    assert (llama / "llama-server.exe").exists()
    assert not (llama / "keep.dll").exists()


@pytest.mark.parametrize("which", ["llama", "old"])
def test_install_llama_refuses_unstamped_dir_and_keeps_it(release, which):
    release()
    llama, staging, old = _dirs()
    target = llama if which == "llama" else old
    write(target / "mine.gguf", "precious")
    with pytest.raises(RuntimeError, match="was not installed by cutpoint setup"):
        setup.install_llama(force=True)
    assert (target / "mine.gguf").read_text() == "precious"
    assert not staging.exists()


def test_install_llama_restores_install_left_aside_by_interrupted_run(release):
    release(cuda_digest="0" * 64)  # the new install fails, so only the restore can explain keep.dll
    llama, staging, old = _dirs()
    write(old / "VERSION.txt", OLD_STAMP)  # earlier run died after moving llama.cpp aside
    write(old / "keep.dll", "old")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        setup.install_llama(force=True)
    _assert_old_kept(llama, staging, old)


def test_install_llama_up_to_date_removes_leftover_old_without_downloading(release, caplog):
    release()
    llama = _old_install()
    _, staging, old = _dirs()
    write(old / "VERSION.txt", OLD_STAMP)  # earlier swap could not delete it (llama-server ran)
    write(old / "stale.dll", "old")
    with caplog.at_level("INFO"):
        setup.install_llama()
    assert "already installed" in caplog.text
    assert not old.exists()
    assert not staging.exists()
    assert not (ev_paths.tools_dir() / "_downloads").exists()
    assert sorted(p.name for p in llama.iterdir()) == ["VERSION.txt", "keep.dll"]
    assert (llama / "keep.dll").read_text() == "old"


# -- #30: PANNs checkpoint mirrors ----------------------------------------------------------------


@pytest.mark.parametrize("first", ["missing", "wrong_sha", "slow"])
def test_install_panns_falls_back_to_next_mirror(tmp_path, monkeypatch, caplog, first):
    good = b"good-checkpoint" * 20
    srv = tmp_path / "srv"
    bad_url = (
        (srv / "gone.pth").as_uri()
        if first == "missing"
        else write_bytes(srv / "bad.pth", b"X" * len(good))  # right size, wrong content
    )
    if first == "slow":  # both mirrors serve the good file; the first is judged too slow
        bad_url = write_bytes(srv / "slow.pth", good)
        monkeypatch.setattr(setup, "SLOW_START_WINDOW", 0.0)
        monkeypatch.setattr(setup, "PANNS_MIN_RATE", float("inf"))
    good_url = write_bytes(srv / "good.pth", good)
    ckpt = tmp_path / "models" / "panns" / "ckpt.pth"
    monkeypatch.setattr(setup, "PANNS_CKPT_MIRRORS", (bad_url, good_url))
    monkeypatch.setattr(setup, "PANNS_CKPT_SIZE", len(good))
    monkeypatch.setattr(setup, "PANNS_CKPT_SHA256", hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(setup, "PANNS_LABELS_URL", write_bytes(srv / "labels.csv", b"a,b\n"))
    monkeypatch.setattr(ev_paths, "panns_ckpt", lambda: ckpt)
    monkeypatch.setattr(ev_paths, "panns_labels", lambda: tmp_path / "models" / "panns" / "l.csv")
    monkeypatch.delenv("CUTPOINT_PANNS_URL", raising=False)

    with caplog.at_level("INFO"):
        setup.install_panns()

    assert ckpt.read_bytes() == good
    # the user can see which mirror the checkpoint finally came from (#52); slow keeps its .part
    verb = "resuming" if first == "slow" else "downloading"
    assert f"{verb} {ckpt.name} from {good_url}" in caplog.text
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert bad_url in warnings[0]
    assert good_url not in warnings[0]
    if first == "slow":  # the last mirror is exempt, so it still completes
        assert "too slow" in warnings[0]


@pytest.mark.parametrize(
    ("bad", "shown"),
    [
        ("http://[foo", "[foo"),
        ("https://user:TOKEN@[foo/x?t=SECRET", "[foo"),
        ("user:TOKEN@host/x?t=SECRET", "host"),  # no "//": parses with no host
        ("http://user:TOKEN@host:SECRET/x", "host"),  # nonnumeric port
        ("TOKEN@host/x?t=SECRET", "host"),  # no scheme: "unknown url type"
        ("TOKEN:x@host/x?t=SECRET", "host"),  # urllib reads "token" as the scheme
        ("file:https://user:TOKEN@host/x?t=SECRET", "host"),
    ],
    ids=[
        "plain",
        "credentials",
        "no-slashes",
        "bad-port",
        "no-scheme",
        "token-user",
        "file-nested",
    ],
)
def test_install_panns_malformed_env_url_is_a_failed_mirror(
    tmp_path, monkeypatch, caplog, bad, shown
):
    good = b"good-checkpoint" * 20
    srv = tmp_path / "srv"
    good_url = write_bytes(srv / "good.pth", good)
    ckpt = tmp_path / "models" / "panns" / "ckpt.pth"
    monkeypatch.setattr(setup, "PANNS_CKPT_MIRRORS", (good_url,))
    monkeypatch.setattr(setup, "PANNS_CKPT_SIZE", len(good))
    monkeypatch.setattr(setup, "PANNS_CKPT_SHA256", hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(setup, "PANNS_LABELS_URL", write_bytes(srv / "labels.csv", b"a,b\n"))
    monkeypatch.setattr(ev_paths, "panns_ckpt", lambda: ckpt)
    monkeypatch.setattr(ev_paths, "panns_labels", lambda: tmp_path / "models" / "panns" / "l.csv")
    monkeypatch.setenv("CUTPOINT_PANNS_URL", bad)

    with caplog.at_level("INFO"):
        setup.install_panns()

    assert ckpt.read_bytes() == good  # the next mirror served it
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert shown in warnings[0]
    assert "token" not in caplog.text.lower()  # urllib lowercases the scheme in its text
    assert "secret" not in caplog.text.lower()
