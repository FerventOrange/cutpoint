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
    panns    Cnn14_DecisionLevelMax checkpoint + AudioSet label CSV -> models/panns
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
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from cutpoint.events import paths
from cutpoint.extras import install_hint

logger = logging.getLogger(__name__)

PANNS_CKPT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
)
PANNS_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
)
HF_URL = "https://huggingface.co/{repo}/resolve/main/{name}"
GH_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
CUDA_FLAVOR = "cuda-12.4"  # matches the cudart-llama-bin-win-cuda-12.4 runtime zip
LINUX_ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}

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


def download(url: str, dst: Path, expected_size: int | None = None) -> Path:
    """Stream ``url`` to ``dst`` with resume support.

    Args:
        url: Source URL (redirects are followed).
        dst: Destination file; a ``.part`` file is used while downloading.
        expected_size: Skip the download if ``dst`` already has this size.

    Returns:
        ``dst``.
    """
    if dst.exists() and (expected_size is None or dst.stat().st_size == expected_size):
        logger.info("have %s", dst.name)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "cutpoint-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored Range: restart
            have = 0
        total = int(resp.headers.get("Content-Length") or 0) + have
        mode = "ab" if have else "wb"
        done, next_log = have, 0.0
        with part.open(mode) as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done / total >= next_log:
                    logger.info("%s %5.1f%% of %.2f GB", dst.name, 100 * done / total, total / 1e9)
                    next_log += 0.1
    part.replace(dst)
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
    """Download the PANNs checkpoint and the AudioSet label CSV."""
    download(PANNS_LABELS_URL, paths.panns_labels())
    download(PANNS_CKPT_URL, paths.panns_ckpt())


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


def _asset(assets: dict[str, tuple[str, int]], stem: str) -> tuple[str, int] | None:
    """``stem`` + ``.tar.gz`` (current Linux releases) or ``.zip`` (older ones)."""
    return assets.get(f"{stem}.tar.gz") or assets.get(f"{stem}.zip")


def _pick_linux_assets(
    tag: str, assets: dict[str, tuple[str, int]], arch: str, nvidia: bool
) -> tuple[str, dict[str, tuple[str, int]]] | None:
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


def _latest_llama_assets(plat: str) -> tuple[str, str, dict[str, tuple[str, int]]]:
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
        assets = {a["name"]: (a["browser_download_url"], a["size"]) for a in rel["assets"]}
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
    stamp = paths.llama_dir() / "VERSION.txt"
    tag, flavor, assets = _latest_llama_assets(plat)
    if stamp.exists() and not force:
        have = stamp.read_text().strip()
        if _stamp_fits(have, plat):
            logger.info(
                "llama.cpp already installed (%s); latest is %s. Use --force to replace.",
                have,
                tag,
            )
            return
        logger.info("llama.cpp in %s is for another platform (%s); replacing", stamp.parent, have)
    dl_root = paths.tools_dir() / "_downloads"
    for key, (url, size) in assets.items():
        folder = dl_root / f"llama-{tag}-{key}"  # one fresh folder per archive
        archive = download(url, folder / url.rsplit("/", 1)[-1], size)
        _extract(archive, paths.llama_dir())
    if plat == "linux":  # only our own build, never a CUTPOINT_LLAMA_SERVER or PATH binary
        for exe in paths.llama_dir().glob("**/llama-server"):
            if exe.is_file():
                exe.chmod(exe.stat().st_mode | 0o111)
    stamp.write_text(f"{tag} {flavor}\n")
    shutil.rmtree(dl_root, ignore_errors=True)
    logger.info("llama.cpp %s (%s) installed in %s", tag, flavor, paths.llama_dir())


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
