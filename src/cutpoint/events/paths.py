"""Filesystem locations used by the events subpackage (models, caches).

Everything is a function evaluated on each call, so environment overrides set
after import (``CUTPOINT_HOME``, ``CUTPOINT_MODELS_DIR``, ...) are honoured. The
roots come from :mod:`cutpoint.paths`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from cutpoint import paths as _paths
from cutpoint import state

AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
# Hugging Face repo open_clip loads ViT-L-14 "openai" from, and torchvision's
# Faster R-CNN v2 COCO checkpoint name (both downloaded by ``setup --only visual``).
CLIP_HF_REPO = "timm/vit_large_patch14_clip_224.openai"
DETECTOR_CKPT = "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"


def panns_dir() -> Path:
    """PANNs checkpoint and label CSV."""
    return _paths.models_dir() / "panns"


def panns_ckpt() -> Path:
    """PANNs Cnn14_DecisionLevelMax checkpoint."""
    return panns_dir() / "Cnn14_DecisionLevelMax_mAP=0.385.pth"


def panns_labels() -> Path:
    """AudioSet class label CSV."""
    return panns_dir() / "class_labels_indices.csv"


def clip_dir() -> Path:
    """open_clip / Hugging Face download cache (CLIP ViT-L/14)."""
    return _paths.models_dir() / "clip"


def clip_weights() -> Path:
    """CLIP ViT-L/14 weights in the Hugging Face cache under :func:`clip_dir`.

    The repo folder (not a file) when they are not downloaded.
    """
    repo = clip_dir() / ("models--" + CLIP_HF_REPO.replace("/", "--"))
    hits = sorted(repo.glob("snapshots/*/open_clip_*model.*"))
    return hits[0] if hits else repo


def torch_home() -> Path:
    """``TORCH_HOME`` for the torchvision detector weights."""
    return _paths.models_dir() / "torch"


# The TORCH_HOME value Cutpoint last set ({"value": path}); see apply_torch_home.
_TORCH_HOME_SET: dict[str, str] = {}


def apply_torch_home() -> None:
    """Point ``TORCH_HOME`` at :func:`torch_home` unless the user set it.

    It is (re)set when unset or still equal to the value Cutpoint set last, so
    a changed ``CUTPOINT_HOME`` / ``CUTPOINT_MODELS_DIR`` is followed and a
    user's own ``TORCH_HOME`` is left alone.
    """
    want = str(torch_home())
    current = os.environ.get("TORCH_HOME")
    if current is None or current == _TORCH_HOME_SET.get("value"):
        os.environ["TORCH_HOME"] = want
        _TORCH_HOME_SET["value"] = want


def _forget_torch_home() -> None:
    """Reset hook: undo our ``TORCH_HOME`` (if still ours) and forget it."""
    ours = _TORCH_HOME_SET.pop("value", None)
    if ours is not None and os.environ.get("TORCH_HOME") == ours:
        del os.environ["TORCH_HOME"]


state.register(
    state.Entry(f"{__name__}._TORCH_HOME_SET", _forget_torch_home, lambda: len(_TORCH_HOME_SET))
)


def detector_weights() -> Path:
    """Faster R-CNN v2 checkpoint in the torch hub cache of the ``TORCH_HOME`` that
    :func:`apply_torch_home` would use (without setting it)."""
    current = os.environ.get("TORCH_HOME")
    home = torch_home() if current in (None, _TORCH_HOME_SET.get("value")) else Path(current)
    return home / "hub" / "checkpoints" / DETECTOR_CKPT


def aesthetic_head() -> Path:
    """LAION improved-aesthetic-predictor head."""
    return _paths.models_dir() / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth"


def silero_vad() -> Path:
    """Silero VAD v5 ONNX model."""
    return _paths.models_dir() / "silero" / "silero_vad.onnx"


def layer_files(layers: Iterable[str]) -> list[tuple[str, Path]]:
    """``(setup component, path)`` of every model file ``layers`` load.

    Without duplicates, in layer order; OCR ships its models in the events extra, and the
    VLM layer uses the shared vision model (Ollama or a hosted Vision role), checked when
    its client is made.
    """
    panns = [("panns", panns_ckpt()), ("panns", panns_labels())]
    clip = [("visual", clip_weights())]
    need = {
        "audio": panns,
        "voice": [*panns, ("voice", silero_vad())],
        "visual": [*clip, ("visual", detector_weights()), ("visual", aesthetic_head())],
        "crash": clip,
    }
    out: list[tuple[str, Path]] = []
    for layer in layers:
        out += [item for item in need.get(layer, []) if item not in out]
    return out


def missing_files(layers: Iterable[str]) -> list[tuple[str, Path]]:
    """The entries of :func:`layer_files` that do not exist."""
    return [(comp, p) for comp, p in layer_files(layers) if not p.is_file()]
