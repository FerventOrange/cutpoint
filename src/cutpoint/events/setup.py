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
    llama    latest llama.cpp Windows CUDA 12.4 build + matching cudart DLLs
             -> tools/llama.cpp
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
import shutil
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


def _latest_llama_assets() -> tuple[str, dict[str, tuple[str, int]]]:
    with urllib.request.urlopen(GH_RELEASES, timeout=60) as resp:
        releases = json.load(resp)
    for rel in releases:
        assets = {a["name"]: (a["browser_download_url"], a["size"]) for a in rel["assets"]}
        bin_zip = f"llama-{rel['tag_name']}-bin-win-{CUDA_FLAVOR}-x64.zip"
        rt_zip = f"cudart-llama-bin-win-{CUDA_FLAVOR}-x64.zip"
        if bin_zip in assets and rt_zip in assets:
            return rel["tag_name"], {"bin": assets[bin_zip], "rt": assets[rt_zip]}
    raise RuntimeError("No recent llama.cpp release with Windows CUDA 12.4 assets")


def install_llama(force: bool = False) -> None:
    """Install the newest llama.cpp Windows CUDA build plus its cudart DLLs."""
    stamp = paths.llama_dir() / "VERSION.txt"
    tag, assets = _latest_llama_assets()
    if stamp.exists() and not force:
        logger.info(
            "llama.cpp already installed (%s); latest is %s. Use --force to replace.",
            stamp.read_text().strip(),
            tag,
        )
        return
    dl_root = paths.tools_dir() / "_downloads"
    for key, (url, size) in assets.items():
        folder = dl_root / f"llama-{tag}-{key}"  # one fresh folder per archive
        archive = download(url, folder / url.rsplit("/", 1)[-1], size)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():  # refuse path traversal
                target = (paths.llama_dir() / member).resolve()
                if not str(target).startswith(str(paths.llama_dir().resolve())):
                    raise RuntimeError(f"unsafe path in {archive.name}: {member}")
            zf.extractall(paths.llama_dir())
    stamp.write_text(f"{tag} {CUDA_FLAVOR}\n")
    shutil.rmtree(dl_root, ignore_errors=True)
    logger.info("llama.cpp %s installed in %s", tag, paths.llama_dir())


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
