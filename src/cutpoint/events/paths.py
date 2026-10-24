"""Filesystem locations used by the events subpackage (models, tools, caches).

Everything is a function evaluated on each call, so environment overrides set
after import (``CUTPOINT_HOME``, ``CUTPOINT_MODELS_DIR``, ...) are honoured. The
roots come from :mod:`cutpoint.paths`.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from cutpoint import paths as _paths
from cutpoint import state

OMNI_HF_REPO = "ggml-org/Qwen2.5-Omni-7B-GGUF"
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


def llm_dir() -> Path:
    """GGUF language models."""
    return _paths.models_dir() / "llm"


def omni_model() -> Path:
    """Qwen2.5-Omni-7B Q4_K_M GGUF."""
    return llm_dir() / "Qwen2.5-Omni-7B-Q4_K_M.gguf"


def omni_mmproj() -> Path:
    """Qwen2.5-Omni-7B Q8_0 multimodal projector."""
    return llm_dir() / "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"


def llama_cache() -> Path:
    """``LLAMA_CACHE`` for ``-hf`` fallback loading."""
    return _paths.models_dir() / "llama-cache"


def llama_dir() -> Path:
    """llama.cpp install folder."""
    return _paths.tools_dir() / "llama.cpp"


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


def llama_server_exe() -> Path:
    """Path of ``llama-server`` (``llama-server.exe`` on Windows).

    ``CUTPOINT_LLAMA_SERVER`` first, then the build ``cutpoint setup`` put in
    ``tools/llama.cpp``, then ``llama-server`` on ``PATH`` (e.g. a self-built
    one). If none exists, the expected path in ``tools/llama.cpp``.
    """
    env = _paths.env_path("LLAMA_SERVER")
    if env:
        return env
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    folder = llama_dir()
    hits = sorted(folder.glob(f"**/{name}"), key=lambda p: len(str(p)))
    if hits:
        return hits[0]
    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else folder / name


def layer_files(layers: Iterable[str]) -> list[tuple[str, Path]]:
    """``(setup component, path)`` of every model file and tool ``layers`` load.

    Without duplicates, in layer order; OCR ships its models in the events extra.
    """
    panns = [("panns", panns_ckpt()), ("panns", panns_labels())]
    clip = [("visual", clip_weights())]
    need = {
        "audio": panns,
        "voice": [*panns, ("voice", silero_vad())],
        "visual": [*clip, ("visual", detector_weights()), ("visual", aesthetic_head())],
        "crash": clip,
        "vlm": [("llama", llama_server_exe()), ("omni", omni_model()), ("omni", omni_mmproj())],
    }
    out: list[tuple[str, Path]] = []
    for layer in layers:
        out += [item for item in need.get(layer, []) if item not in out]
    return out


def missing_files(layers: Iterable[str]) -> list[tuple[str, Path]]:
    """The entries of :func:`layer_files` that do not exist."""
    return [(comp, p) for comp, p in layer_files(layers) if not p.is_file()]
