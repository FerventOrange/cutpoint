"""Download and install everything ``cutpoint events`` needs.

Usage::

    cutpoint setup                        # everything
    cutpoint setup --only panns voice

The Python packages (torch, torchvision, panns-inference, ...) are not installed
here: they are the ``events`` extra of the package, so their versions live in
one place (``pyproject.toml``). Cutpoint lite leaves them out. A clone's
``uv sync`` installs them (unless ``--no-group full``), a tool install gets them
from the installer when it is not lite; torch then comes from the PyTorch CUDA
12.6 index on Windows and Linux when there is an NVIDIA GPU.

Components (models go to CUTPOINT_HOME, see ``cutpoint paths``):
    pip      check that the ``events`` extra is installed; print the install
             command when it is not (installs nothing)
    panns    Cnn14_DecisionLevelMax checkpoint + AudioSet label CSV -> models/panns.
             The checkpoint comes from the first mirror that works (see
             PANNS_CKPT_MIRRORS); set CUTPOINT_PANNS_URL to try your own URL first.
    visual   CLIP ViT-L/14 (OpenAI weights, ~1.7 GB) -> models/clip, Faster R-CNN
             ResNet50-FPN v2 COCO (167 MB) -> models/torch, LAION aesthetic
             head (3.7 MB) -> models/aesthetic
    voice    Silero VAD v5 ONNX (2.3 MB) -> models/silero
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from cutpoint import paths as cutpoint_paths
from cutpoint.events import paths
from cutpoint.extras import LITE, install_hint

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
# LAION improved-aesthetic-predictor head (Apache-2.0), converted from the upstream pickled
# .pth to .npz by scripts/convert_aesthetic_head.py (float32 values unchanged); tried in order.
AESTHETIC_MIRRORS: tuple[str, ...] = (
    "https://huggingface.co/FerventOrange/cutpoint-models/resolve/main/aesthetic/"
    "sac%2Blogos%2Bava1-l14-linearMSE.npz",
    "https://github.com/FerventOrange/cutpoint/releases/download/models-aesthetic-v1/"
    "aesthetic-head-v1.npz",
)
AESTHETIC_SIZE = 3714498
AESTHETIC_SHA256 = "8aa479a5065da0ad9941938da7f5ef8eeb66df8bf91385a8381557ee511f3770"
# A mirror with another one after it is dropped if it delivers less than this (bytes/s)
# over its first SLOW_START_WINDOW seconds. The last mirror is never dropped for slowness.
PANNS_MIN_RATE = 500_000
SLOW_START_WINDOW = 20.0
PANNS_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
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
)
COMPONENTS = ("pip", "panns", "visual", "voice")


class SlowDownloadError(Exception):
    """A download started slower than the ``min_rate`` passed to :func:`download`."""


def _sha256(path: Path) -> str:
    """Hex sha256 of ``path``, read in chunks."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _host(url: str) -> str:
    """Host name of ``url`` for log lines (no credentials, port, path or query).

    A local ``file:`` URL (no host, absolute path) is shown as ``file://<path>``: a local path
    carries no credentials and is what the user needs to see. Any other value
    without a host name (malformed, or parsed with no host such as
    ``user:TOKEN@host/x``) is reduced to its authority: no scheme, path, query,
    fragment, text up to the last ``@`` or port. Nothing left gives a placeholder.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        parts = None
    if parts is not None:
        if parts.hostname:
            return parts.hostname
        if parts.scheme == "file" and not parts.netloc and parts.path.startswith("/"):
            return f"file://{parts.path}"
    authority = re.split(r"[/?#]", url.split("://", 1)[-1], maxsplit=1)[0].rpartition("@")[2]
    if authority.startswith("["):  # IPv6 literal, possibly unclosed: drop a port after "]"
        host = authority.partition("]")[0] + ("]" if "]" in authority else "")
    else:
        host = authority.partition(":")[0]
    return host or "<no host>"


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
    logger.info("%s %s from %s", "resuming" if have else "downloading", dst.name, _host(url))
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
        "Missing Python packages for cutpoint events: %s; cutpoint events is %s. To add "
        "the events extra: %s",
        ", ".join(missing),
        LITE,
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
        host = _host(url)
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
                logger.warning("%s: sha256 mismatch after resuming; restarting", dst.name)
                return download(url, dst, size, sha256, min_rate=min_rate)
        except (
            OSError,  # includes URLError and timeouts
            http.client.HTTPException,
            ValueError,  # malformed URL
            RuntimeError,  # sha256 mismatch
            SlowDownloadError,
        ) as e:
            # urllib quotes parts of a bad URL (credentials, port, query, a token as
            # the "scheme", a non-local file: path) in these messages, so only the
            # type is shown; others get the URL redacted.
            if (
                isinstance(e, (ValueError, http.client.InvalidURL))
                or (
                    isinstance(e, urllib.error.URLError)
                    and str(e.reason).startswith("unknown url type")
                )
                or (url.lower().startswith("file:") and not host.startswith("file://"))
            ):
                reason = type(e).__name__
            else:
                reason = str(e).replace(url, host)
            if isinstance(e, SlowDownloadError):
                reason = f"too slow: {reason}"
            nxt = "; trying the next one" if i < len(urls) - 1 else ""
            logger.warning("%s: mirror %s failed (%s)%s", dst.name, host, reason, nxt)
            failures.append(f"{host}: {reason}")
    raise RuntimeError(f"Could not download {dst.name} from any mirror: " + "; ".join(failures))


def nvidia_usable() -> bool:
    """True if an NVIDIA driver answers: Linux kernel driver loaded, or ``nvidia-smi -L`` works."""
    if Path("/proc/driver/nvidia/version").exists():
        return True
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        proc = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout


def install_visual() -> None:
    """Download the aesthetic head; load CLIP + Faster R-CNN once so their weights download.

    The head comes from the first of ``AESTHETIC_MIRRORS`` that works; the pickled ``.pth``
    earlier versions downloaded is then removed.
    """
    head = paths.aesthetic_head()
    _download_mirrors(list(AESTHETIC_MIRRORS), head, AESTHETIC_SIZE, AESTHETIC_SHA256)
    old = head.with_suffix(".pth")
    for f in (old, old.with_name(old.name + ".part")):
        f.unlink(missing_ok=True)
    for m in ("torch", "open_clip"):
        if importlib.util.find_spec(m) is None:
            # The CLI turns this into "<m> is not installed: ... not available in Cutpoint lite"
            raise ModuleNotFoundError(f"No module named {m!r}", name=m)
    from cutpoint.events.visual import VisualModels  # noqa: PLC0415

    VisualModels(download=True).close()  # the only CLIP load that may download


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


def run_args(args: argparse.Namespace) -> int:
    """Run the setup steps selected in ``args``; returns a process exit code."""
    logger.info("models -> %s", cutpoint_paths.models_dir())
    steps = {
        "pip": check_python_packages,
        "panns": install_panns,
        "visual": install_visual,
        "voice": install_voice,
    }
    for name in args.only:
        logger.info("== %s", name)
        steps[name]()
    return 0
