"""Filesystem locations used by the events subpackage (models, tools, caches).

Everything is a function evaluated on each call, so environment overrides set
after import (``CUTPOINT_HOME``, ``CUTPOINT_MODELS_DIR``, ...) are honoured. The
roots come from :mod:`cutpoint.paths`.
"""

from __future__ import annotations

import os
from pathlib import Path

from cutpoint import paths as _paths
from cutpoint import state

OMNI_HF_REPO = "ggml-org/Qwen2.5-Omni-7B-GGUF"
AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def models_dir() -> Path:
    """Model weights root (``CUTPOINT_HOME/models`` unless overridden)."""
    return _paths.models_dir()


def tools_dir() -> Path:
    """External tools root (``CUTPOINT_HOME/tools`` unless overridden)."""
    return _paths.tools_dir()


def panns_dir() -> Path:
    """PANNs checkpoint and label CSV."""
    return models_dir() / "panns"


def panns_ckpt() -> Path:
    """PANNs Cnn14_DecisionLevelMax checkpoint."""
    return panns_dir() / "Cnn14_DecisionLevelMax_mAP=0.385.pth"


def panns_labels() -> Path:
    """AudioSet class label CSV."""
    return panns_dir() / "class_labels_indices.csv"


def llm_dir() -> Path:
    """GGUF language models."""
    return models_dir() / "llm"


def omni_model() -> Path:
    """Qwen2.5-Omni-7B Q4_K_M GGUF."""
    return llm_dir() / "Qwen2.5-Omni-7B-Q4_K_M.gguf"


def omni_mmproj() -> Path:
    """Qwen2.5-Omni-7B Q8_0 multimodal projector."""
    return llm_dir() / "mmproj-Qwen2.5-Omni-7B-Q8_0.gguf"


def llama_cache() -> Path:
    """``LLAMA_CACHE`` for ``-hf`` fallback loading."""
    return models_dir() / "llama-cache"


def llama_dir() -> Path:
    """llama.cpp install folder."""
    return tools_dir() / "llama.cpp"


def clip_dir() -> Path:
    """open_clip / Hugging Face download cache (CLIP ViT-L/14)."""
    return models_dir() / "clip"


def torch_home() -> Path:
    """``TORCH_HOME`` for the torchvision detector weights."""
    return models_dir() / "torch"


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


def aesthetic_head() -> Path:
    """LAION improved-aesthetic-predictor head."""
    return models_dir() / "aesthetic" / "sac+logos+ava1-l14-linearMSE.pth"


def silero_vad() -> Path:
    """Silero VAD v5 ONNX model."""
    return models_dir() / "silero" / "silero_vad.onnx"


def llama_server_exe() -> Path:
    """Path of ``llama-server.exe`` (``CUTPOINT_LLAMA_SERVER`` overrides it)."""
    env = _paths.env_path("LLAMA_SERVER")
    if env:
        return env
    folder = llama_dir()
    hits = sorted(folder.glob("**/llama-server.exe"), key=lambda p: len(str(p)))
    return hits[0] if hits else folder / "llama-server.exe"
