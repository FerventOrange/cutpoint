"""Download and install everything ``cutpoint events`` needs.

Usage::

    cutpoint setup                        # everything
    cutpoint setup --only panns voice

The Python packages (torch, torchvision, panns-inference, ...) are not installed
here: they are the ``events`` extra of the package, so their versions live in
one place (``pyproject.toml``). Install them with
``uv sync --extra events`` (or ``uv pip install -e ".[events]"`` /
``pip install -e ".[events]"``); torch then comes from the PyTorch CUDA 12.6
index on Windows and Linux.

Components (models and tools go to CUTPOINT_HOME, see ``cutpoint paths``):
    pip      check that the ``events`` extra is installed; print the install
             command when it is not (installs nothing)
    panns    Cnn14_DecisionLevelMax checkpoint + AudioSet label CSV -> models/panns.
             The checkpoint comes from the first mirror that works (see
             PANNS_CKPT_MIRRORS); set CUTPOINT_PANNS_URL to try your own URL first.
    llama    latest llama.cpp build -> tools/llama.cpp. Windows: CUDA 12.4
             build + matching cudart DLLs. Linux: CUDA build + cudart libraries
             when an NVIDIA driver is present, else the Vulkan build (NVIDIA,
             AMD, Intel), else the CPU build. Other platforms: put your own
             llama-server on PATH or set CUTPOINT_LLAMA_SERVER.
    omni     Qwen2.5-Omni-7B Q4_K_M + mmproj Q8_0 -> models/llm   (~6.2 GB)
    visual   CLIP ViT-L/14 (OpenAI weights, ~1.7 GB) -> models/clip, Faster R-CNN
             ResNet50-FPN v2 COCO (167 MB) -> models/torch, LAION aesthetic
             head (3.7 MB) -> models/aesthetic
    voice    Silero VAD v5 ONNX (2.3 MB) -> models/silero

Every archive is downloaded into its own fresh folder under
``tools/_downloads`` before being extracted. Downloads resume if interrupted.
llama.cpp archives are checked against the sha256 digest GitHub publishes
(when the release has one), extracted into ``tools/llama.cpp.new`` and then
swapped in for ``tools/llama.cpp``, so files of an older build never linger.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from cutpoint.events import paths
from cutpoint.extras import install_hint

logger = logging.getLogger(__name__)

# PANNs checkpoint mirrors, tried in order (CUTPOINT_PANNS_URL, if set, comes first).
# Zenodo is often very slow (~0.1 MB/s).
PANNS_CKPT_MIRRORS: tuple[str, ...] = (
    # Hugging Face copy of the Zenodo record (CC BY 4.0); Zenodo is the original.
    "https://huggingface.co/FerventOrange/cutpoint-models/resolve/main/panns/"
    "Cnn14_DecisionLevelMax_mAP%3D0.385.pth",
    "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1",
)
PANNS_CKPT_SIZE = 327428481
PANNS_CKPT_SHA256 = "dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142"
# A mirror with another one after it is dropped if it delivers less than this (bytes/s)
# over its first SLOW_START_WINDOW seconds. The last mirror is never dropped for slowness.
PANNS_MIN_RATE = 500_000
SLOW_START_WINDOW = 20.0
PANNS_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
)
HF_URL = "https://huggingface.co/{repo}/resolve/main/{name}"
GH_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
CUDA_FLAVOR = "cuda-12.4"  # matches the cudart-llama-bin-win-cuda-12.4 runtime zip
LINUX_ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
_OLD_KEPT = (
    "Could not remove %s (a running llama-server holds files in it); it is removed on the "
    "next cutpoint setup --only llama"
)

# Import names of the ``events`` extra (see pyproject.toml).
EVENTS_MODULES = (
    "torch",
    "torchvision",
    "panns_inference",
    "torchlibrosa",
    "rapidocr_onnxruntime",
    "open_clip",
    "onnxruntime",
    "scipy",
    "sklearn",
)
COMPONENTS = ("pip", "panns", "llama", "omni", "visual", "voice")

# Release asset: (download URL, size in bytes, sha256 hex digest or None).
Asset = tuple[str, int, str | None]


class SlowDownloadError(Exception):
    """A download started slower than the ``min_rate`` passed to :func:`download`."""


def _sha256(path: Path) -> str:
    """Hex sha256 of ``path``, read in chunks."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def download(
    url: str,
    dst: Path,
    expected_size: int | None = None,
    sha256: str | None = None,
    min_rate: float | None = None,
) -> Path:
    """Stream ``url`` to ``dst`` with resume support.

    Args:
        url: Source URL (redirects are followed).
        dst: Destination file; a ``.part`` file is used while downloading.
        expected_size: Skip the download if ``dst`` already has this size.
        sha256: Expected hex digest; an existing ``dst`` that does not match is
            downloaded again, a fresh download that does not match is deleted.
        min_rate: Minimum speed in bytes/s, checked once after
            ``SLOW_START_WINDOW`` seconds; None never checks. The ``.part`` file
            is kept, so another URL serving the same file can resume it.

    Returns:
        ``dst``.

    Raises:
        RuntimeError: The downloaded file does not match ``sha256``.
        SlowDownloadError: The download is slower than ``min_rate``.
    """
    sha256 = sha256.lower() if sha256 else None
    part = dst.with_name(dst.name + ".part")
    if dst.exists() and (expected_size is None or dst.stat().st_size == expected_size):
        if sha256 is None or _sha256(dst) == sha256:
            logger.info("have %s", dst.name)
            return dst
        logger.warning("%s does not match its sha256; downloading it again", dst.name)
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "cutpoint-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored Range: restart
            have = 0
        total = int(resp.headers.get("Content-Length") or 0) + have
        mode = "ab" if have else "wb"
        done, next_log = have, 0.0
        first = done
        with part.open(mode) as fh:
            while chunk := resp.read1(1 << 20):  # returns what has arrived
                fh.write(chunk)
                done += len(chunk)
                if min_rate is not None and (
                    (elapsed := time.monotonic() - started) >= SLOW_START_WINDOW
                ):
                    if (rate := (done - first) / max(elapsed, 1e-9)) < min_rate:
                        raise SlowDownloadError(
                            f"{rate / 1e6:.2f} MB/s after {elapsed:.0f} s "
                            f"(minimum {min_rate / 1e6:.2f} MB/s)"
                        )
                    min_rate = None  # checked once
                if total and done / total >= next_log:
                    logger.info("%s %5.1f%% of %.2f GB", dst.name, 100 * done / total, total / 1e9)
                    next_log += 0.1
    part.replace(dst)
    if sha256 is not None and (actual := _sha256(dst)) != sha256:
        dst.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{dst.name}: sha256 mismatch (expected {sha256}, got {actual})")
    return dst


def check_python_packages() -> None:
    """Report whether the ``events`` extra is installed (installs nothing)."""
    missing = [m for m in EVENTS_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        logger.info("events extra installed (%s)", ", ".join(EVENTS_MODULES))
        return
    logger.warning(
        "Missing Python packages for cutpoint events: %s. Install the events extra: %s",
        ", ".join(missing),
        install_hint("events"),
    )


def install_panns() -> None:
    """Download the PANNs checkpoint and the AudioSet label CSV.

    The checkpoint comes from the first of ``PANNS_CKPT_MIRRORS`` that works,
    preceded by ``CUTPOINT_PANNS_URL`` when that environment variable is set.
    """
    download(PANNS_LABELS_URL, paths.panns_labels())
    urls = list(PANNS_CKPT_MIRRORS)
    if env_url := os.environ.get("CUTPOINT_PANNS_URL", "").strip():
        urls = [env_url, *(u for u in urls if u != env_url)]
    _download_mirrors(urls, paths.panns_ckpt(), PANNS_CKPT_SIZE, PANNS_CKPT_SHA256)


def _download_mirrors(urls: list[str], dst: Path, size: int, sha256: str) -> Path:
    """Download ``dst`` from the first of ``urls`` that works.

    A mirror that fails (network error, sha256 mismatch) or, unless it is the
    last one, starts slower than ``PANNS_MIN_RATE`` is skipped with a warning.
    A partial download stays as ``.part`` for the next mirror to resume; the
    sha256 check at the end rejects a mixed-up file (and deletes it). A mirror
    that resumed an existing ``.part`` and fails that check is retried once from
    scratch, so bytes from an earlier mirror are not blamed on it.

    Raises:
        RuntimeError: Every mirror failed; lists each one with its reason.
    """
    failures = []
    for i, url in enumerate(urls):
        host = urllib.parse.urlsplit(url).netloc or url
        min_rate = PANNS_MIN_RATE if i < len(urls) - 1 else None
        part = dst.with_name(dst.name + ".part")
        try:
            try:
                had_part = part.exists()
                return download(url, dst, size, sha256, min_rate=min_rate)
            except RuntimeError:
                if not had_part:
                    raise
                # The .part (from an earlier mirror) may have been the bad bytes;
                # download() deleted it, so retry this mirror once from scratch.
                logger.warning("%s: sha256 mismatch after resuming; retrying %s", dst.name, host)
                return download(url, dst, size, sha256, min_rate=min_rate)
        except (
            OSError,  # includes URLError and timeouts
            http.client.HTTPException,
            ValueError,  # malformed URL
            RuntimeError,  # sha256 mismatch
            SlowDownloadError,
        ) as e:
            reason = f"too slow: {e}" if isinstance(e, SlowDownloadError) else str(e)
            nxt = "; trying the next one" if i < len(urls) - 1 else ""
            logger.warning("%s: mirror %s failed (%s)%s", dst.name, host, reason, nxt)
            failures.append(f"{host}: {reason}")
    raise RuntimeError(f"Could not download {dst.name} from any mirror: " + "; ".join(failures))


def _llama_platform() -> str:
    """``"win"`` or ``"linux"``; other platforms have no prebuilt install here."""
    if os.name == "nt":
        return "win"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(
        f"cutpoint setup installs llama.cpp on Windows and Linux only ({sys.platform}); "
        "put your own llama-server on PATH or set CUTPOINT_LLAMA_SERVER"
    )


def _has_nvidia() -> bool:
    """True if an NVIDIA driver is loaded (Linux)."""
    return Path("/proc/driver/nvidia/version").exists() or shutil.which("nvidia-smi") is not None


def _asset(assets: dict[str, Asset], stem: str) -> Asset | None:
    """``stem`` + ``.tar.gz`` (current Linux releases) or ``.zip`` (older ones)."""
    return assets.get(f"{stem}.tar.gz") or assets.get(f"{stem}.zip")


def _pick_linux_assets(
    tag: str, assets: dict[str, Asset], arch: str, nvidia: bool
) -> tuple[str, dict[str, Asset]] | None:
    """Best Linux build in one release as ``(flavor, {"bin": ..., ["rt": ...]})``.

    CUDA + its cudart libraries (NVIDIA only; the oldest CUDA version, which
    needs the oldest driver) > Vulkan > CPU. None if the release has none.
    """
    pre = f"llama-{tag}-bin-ubuntu-"
    if nvidia:
        pattern = re.compile(rf"{re.escape(pre)}cuda-(\d+(?:\.\d+)*)-{arch}\.(?:tar\.gz|zip)")
        versions = {m[1] for name in assets if (m := pattern.fullmatch(name))}
        for ver in sorted(versions, key=lambda v: tuple(int(x) for x in v.split("."))):
            bin_ = _asset(assets, f"{pre}cuda-{ver}-{arch}")
            rt = _asset(assets, f"cudart-llama-{tag}-bin-ubuntu-cuda-{ver}-{arch}")
            if bin_ and rt:
                return f"linux-cuda-{ver}", {"bin": bin_, "rt": rt}
    for flavor, stem in (("linux-vulkan", f"{pre}vulkan-{arch}"), ("linux-cpu", f"{pre}{arch}")):
        if bin_ := _asset(assets, stem):
            return flavor, {"bin": bin_}
    return None


def _sha256_of(asset: dict) -> str | None:
    """Hex digest from a release asset's ``"digest": "sha256:<hex>"`` (None if absent)."""
    digest = asset.get("digest") or ""
    return digest.removeprefix("sha256:") if digest.startswith("sha256:") else None


def _latest_llama_assets(plat: str) -> tuple[str, str, dict[str, Asset]]:
    """``(tag, flavor, assets)`` of the newest release with a build for ``plat``."""
    with urllib.request.urlopen(GH_RELEASES, timeout=60) as resp:
        releases = json.load(resp)
    if plat == "linux":
        arch = LINUX_ARCH.get(platform.machine().lower())
        if arch is None:
            raise RuntimeError(f"No llama.cpp Linux build for {platform.machine()}")
        nvidia = _has_nvidia()
    for rel in releases:
        tag = rel["tag_name"]
        assets = {
            a["name"]: (a["browser_download_url"], a["size"], _sha256_of(a)) for a in rel["assets"]
        }
        if plat == "linux":
            if picked := _pick_linux_assets(tag, assets, arch, nvidia):
                return tag, picked[0], picked[1]
            continue
        bin_zip = f"llama-{tag}-bin-win-{CUDA_FLAVOR}-x64.zip"
        rt_zip = f"cudart-llama-bin-win-{CUDA_FLAVOR}-x64.zip"
        if bin_zip in assets and rt_zip in assets:
            return tag, CUDA_FLAVOR, {"bin": assets[bin_zip], "rt": assets[rt_zip]}
    if plat == "linux":
        raise RuntimeError("No recent llama.cpp release with Linux assets")
    raise RuntimeError("No recent llama.cpp release with Windows CUDA 12.4 assets")


def _stamp_fits(stamp_text: str, plat: str) -> bool:
    """True if VERSION.txt (``"<tag> <flavor>"``) is an install for ``plat``.

    Linux flavors start with ``linux-``; Windows stamps are ``<tag> cuda-12.4``.
    """
    parts = stamp_text.split()
    linux = len(parts) > 1 and parts[1].startswith("linux-")
    return linux == (plat == "linux")


def _extract(archive: Path, dest: Path) -> None:
    """Extract a release archive into ``dest``.

    Tarballs (Linux) wrap everything in one top-level folder, which is dropped
    so the binaries and the cudart libraries end up side by side. Zip files
    lose their mode bits in ``zipfile``, so on POSIX they are restored from
    ``external_attr``.
    """
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            tops = {m.name.split("/", 1)[0] for m in members}
            if len(tops) == 1 and all("/" in m.name or m.isdir() for m in members):
                members = [m for m in members if "/" in m.name]
                for m in members:
                    m.name = m.name.split("/", 1)[1]
            tf.extractall(dest, members=members, filter="data")  # refuses path traversal
        return
    with zipfile.ZipFile(archive) as zf:
        root = dest.resolve()
        for member in zf.namelist():  # refuse path traversal
            if not (dest / member).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive.name}: {member}")
        if os.name == "nt":
            zf.extractall(dest)
            return
        for info in zf.infolist():
            out = Path(zf.extract(info, dest))
            if not info.is_dir() and (mode := (info.external_attr >> 16) & 0o777):
                out.chmod(mode & 0o755)


def install_llama(force: bool = False) -> None:
    """Install the newest llama.cpp build for this platform (see the module docstring)."""
    plat = _llama_platform()
    llama = paths.llama_dir()
    stamp = llama / "VERSION.txt"
    # Siblings of llama_dir, so llama_server_exe() never finds a binary in them.
    staging = llama.with_name(llama.name + ".new")
    old = llama.with_name(llama.name + ".old")
    tag, flavor, assets = _latest_llama_assets(plat)
    if old.exists() and not llama.exists() and _ours(old):  # interrupted between the renames
        old.rename(llama)
    have = stamp.read_text().strip() if stamp.exists() and not force else None
    if have is not None and _stamp_fits(have, plat):
        _clean_leftovers(llama, staging, old, strict=False)
        logger.info(
            "llama.cpp already installed (%s); latest is %s. Use --force to replace.",
            have,
            tag,
        )
        return
    if have is not None:
        logger.info("llama.cpp in %s is for another platform (%s); replacing", llama, have)
    _clean_leftovers(llama, staging, old, strict=True)
    staging.mkdir(parents=True)
    dl_root = paths.tools_dir() / "_downloads"
    try:
        for key, (url, size, sha256) in assets.items():
            name = url.rsplit("/", 1)[-1]
            if sha256 is None:
                logger.debug("no sha256 published for %s; not verified", name)
            folder = dl_root / f"llama-{tag}-{key}"  # one fresh folder per archive
            archive = download(url, folder / name, size, sha256)
            _extract(archive, staging)
        if plat == "linux":  # only our own build, never a CUTPOINT_LLAMA_SERVER or PATH binary
            for exe in staging.glob("**/llama-server"):
                if exe.is_file():
                    exe.chmod(exe.stat().st_mode | 0o111)
        (staging / stamp.name).write_text(f"{tag} {flavor}\n")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _swap_in(staging, llama, old)
    shutil.rmtree(dl_root, ignore_errors=True)
    logger.info("llama.cpp %s (%s) installed in %s", tag, flavor, llama)


def _clean_leftovers(llama: Path, staging: Path, old: Path, strict: bool) -> None:
    """Remove ``staging`` and ``old`` left by an earlier run.

    Folders without our VERSION.txt are never deleted. With ``strict`` (before an install)
    such a folder, or an ``old`` that cannot be removed, raises; otherwise both are left
    as they are, with a warning for ``old``.

    Raises:
        RuntimeError: ``strict`` and a folder is not ours or ``old`` stays.
    """
    for d in (old, llama):  # never delete a llama.cpp folder someone else put there
        if d.exists() and not _ours(d):
            if not strict:
                return
            raise RuntimeError(
                f"{d} was not installed by cutpoint setup (no VERSION.txt); move it "
                "elsewhere and run cutpoint setup --only llama again"
            )
    if staging.exists():
        logger.info("removing %s left by an earlier run", staging)
        shutil.rmtree(staging, ignore_errors=not strict)
    if old.exists():
        logger.info("removing %s left by an earlier run", old)
        _rmtree_ours(old)
        if old.exists():
            if strict:
                raise RuntimeError(
                    f"Could not remove {old}. Close whatever holds files in it (e.g. "
                    "llama-server) and run cutpoint setup --only llama again."
                )
            logger.warning(_OLD_KEPT, old)


def _ours(d: Path) -> bool:
    """True if ``d`` is missing, has our VERSION.txt or holds only llama-server.log."""
    if not d.exists():
        return True
    return (d / "VERSION.txt").exists() or all(p.name == "llama-server.log" for p in d.iterdir())


def _rmtree_ours(d: Path) -> None:
    """Best-effort delete of ``d``, removing its VERSION.txt last.

    If a file stays locked (e.g. Windows AV), the stamp stays too, so ``_ours(d)``
    still holds on the next run.
    """
    if not d.exists():
        return
    stamp = d / "VERSION.txt"
    for p in d.iterdir():
        if p == stamp:
            continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            pass
    try:
        if all(p == stamp for p in d.iterdir()):
            stamp.unlink(missing_ok=True)
            d.rmdir()
    except OSError:
        pass


def _rename_retry(src: Path, dst: Path, tries: int = 5) -> None:
    """``src.rename(dst)``, retried while Windows AV/indexers briefly lock new files."""
    for i in range(tries):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def _swap_in(staging: Path, llama: Path, old: Path) -> None:
    """Replace ``llama`` by ``staging``; the previous install goes via ``old`` and is deleted.

    Raises:
        RuntimeError: A rename failed (e.g. a file locked by antivirus); the previous
            install is restored and ``staging`` removed.
    """
    moved = False
    try:
        if llama.exists():
            _rename_retry(llama, old)
            moved = True
        _rename_retry(staging, llama)
    except OSError as e:
        if moved and not llama.exists():
            old.rename(llama)
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"Could not replace {llama} ({e}). Stop cutpoint and any llama-server "
            "that is running, then run cutpoint setup --only llama again."
        ) from e
    _rmtree_ours(old)
    if old.exists():  # Windows lets a running llama-server's folder be renamed, not deleted
        logger.warning(_OLD_KEPT, old)


def install_omni() -> None:
    """Download the Qwen2.5-Omni-7B Q4_K_M model and its Q8_0 mmproj."""
    for dst in (paths.omni_mmproj(), paths.omni_model()):
        url = HF_URL.format(repo=paths.OMNI_HF_REPO, name=dst.name)
        download(url, dst)


def install_visual() -> None:
    """Download the aesthetic head; load CLIP + Faster R-CNN once so their weights download."""
    download(paths.AESTHETIC_URL, paths.aesthetic_head())
    if any(importlib.util.find_spec(m) is None for m in ("torch", "open_clip")):
        raise RuntimeError(
            "torch / open_clip_torch are not installed; install the events "
            f"extra first: {install_hint('events')}"
        )
    from cutpoint.events.visual import VisualModels  # noqa: PLC0415

    VisualModels().close()


def install_voice() -> None:
    """Download the Silero VAD ONNX model."""
    download(paths.SILERO_URL, paths.silero_vad())


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Add the setup options to ``ap`` (``cutpoint setup``)."""
    ap.add_argument(
        "--only",
        nargs="+",
        choices=COMPONENTS,
        default=list(COMPONENTS),
        help="components to install (default: all)",
    )
    ap.add_argument(
        "--torch-cuda",
        action="store_true",
        help="no longer used: torch comes from the events extra (CUDA 12.6 "
        "on Windows/Linux); kept so old commands still run",
    )
    ap.add_argument("--force", action="store_true", help="Reinstall llama.cpp.")


def run_args(args: argparse.Namespace) -> int:
    """Run the setup steps selected in ``args``; returns a process exit code."""
    logger.info("models -> %s", paths.models_dir())
    logger.info("tools  -> %s", paths.tools_dir())
    steps = {
        "pip": check_python_packages,
        "panns": install_panns,
        "llama": lambda: install_llama(args.force),
        "omni": install_omni,
        "visual": install_visual,
        "voice": install_voice,
    }
    for name in args.only:
        logger.info("== %s", name)
        steps[name]()
    return 0
