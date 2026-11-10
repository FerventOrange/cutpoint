"""CLIP loads from local files only outside ``cutpoint setup`` (#642): no Hub call, no download."""

from __future__ import annotations

import os
import socket
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import messages, write
from cutpoint import actioncam
from cutpoint.crash import ClipEncoder
from cutpoint.events import paths, setup, visual


class _Network(AssertionError):
    """Raised by every patched network entry point."""


class _Loaded(Exception):
    """Raised by the stub open_clip: the model load was reached."""


def _no_network(*_a: Any, **_kw: Any) -> Any:
    raise _Network("network call during a CLIP load")


@pytest.fixture
def no_network(monkeypatch):
    """Every way a CLIP load could reach the network raises :class:`_Network`."""
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    try:
        import requests  # noqa: PLC0415

        monkeypatch.setattr(requests.Session, "request", _no_network)
    except ImportError:
        pass
    try:
        import huggingface_hub  # noqa: PLC0415
        from huggingface_hub import file_download  # noqa: PLC0415

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
        monkeypatch.setattr(file_download, "hf_hub_download", _no_network)
    except ImportError:
        pass


def _fake_weights() -> Path:
    """A fake checkpoint where ``cutpoint setup`` leaves the CLIP weights (HF cache layout)."""
    repo = paths.clip_dir() / ("models--" + paths.CLIP_HF_REPO.replace("/", "--"))
    return write(repo / "snapshots" / "abc123" / "open_clip_model.safetensors", b"fake")


@pytest.fixture
def stub_open_clip(monkeypatch):
    """Stub open_clip, torch and torchvision; the open_clip load records its arguments."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create_model_and_transforms(*a, **kw):
        calls.append((a, kw))
        raise _Loaded

    open_clip = types.SimpleNamespace(create_model_and_transforms=create_model_and_transforms)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), float16="f16", float32="f32"
    )
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace())
    return calls


LOADS = pytest.mark.parametrize(
    "load",
    [
        lambda: ClipEncoder(),
        lambda: visual.VisualModels(),
        lambda: visual.text_embeddings(("a dog",)),
    ],
    ids=["ClipEncoder", "VisualModels", "text_embeddings"],
)


def _repo() -> Path:
    return paths.clip_dir() / ("models--" + paths.CLIP_HF_REPO.replace("/", "--"))


def _snapshot(name: str, mtime: float, *, complete: bool = True) -> Path:
    """A snapshot folder ``name`` (modified at ``mtime``); ``complete`` holds the checkpoint,
    else only the config (a partial download). Returns the checkpoint path."""
    snap = _repo() / "snapshots" / name
    write(snap / "open_clip_config.json", "{}")
    ckpt = snap / "open_clip_model.safetensors"
    if complete:
        write(ckpt, b"fake")
    os.utime(snap, (mtime, mtime))
    return ckpt


def _ref_main(commit: str) -> None:
    write(_repo() / "refs" / "main", commit + "\n")


def test_clip_weights_follow_refs_main_over_sort_order(no_network):
    _snapshot("aaa", 3000.0)  # sorts first and is newest
    b = _snapshot("bbb", 1000.0)
    _ref_main("bbb")
    assert paths.clip_weights() == b


@pytest.mark.parametrize("target", ["missing", "partial"])
def test_clip_weights_bad_ref_falls_back_to_newest_complete(no_network, target):
    _snapshot("aaa", 1000.0)
    newest = _snapshot("ccc", 2000.0)
    _snapshot("ddd", 3000.0, complete=False)  # newer, but without the checkpoint
    if target == "partial":
        _snapshot("eee", 4000.0, complete=False)
    _ref_main("eee")
    assert paths.clip_weights() == newest


def test_clip_weights_without_ref_use_newest_complete(no_network):
    _snapshot("aaa", 1000.0)
    newest = _snapshot("bbb", 2000.0)
    _snapshot("ccc", 3000.0, complete=False)
    assert not (_repo() / "refs").exists()
    assert paths.clip_weights() == newest


def test_clip_weights_ref_is_not_a_path(no_network):
    write(_repo() / "open_clip_model.safetensors", b"stray")  # outside every snapshot
    newest = _snapshot("aaa", 1000.0)
    _ref_main("..")
    assert paths.clip_weights() == newest


def test_clip_weights_ref_not_utf8_falls_back_to_newest_complete(no_network):
    _snapshot("aaa", 1000.0)
    newest = _snapshot("bbb", 2000.0)
    write(_repo() / "refs" / "main", b"\xff\xfe")
    assert paths.clip_weights() == newest


def test_nothing_complete_returns_the_repo_and_raises(stub_open_clip, no_network):
    _snapshot("aaa", 1000.0, complete=False)
    _ref_main("aaa")
    assert paths.clip_weights() == _repo()
    with pytest.raises(visual.ClipWeightsMissing, match="cutpoint setup --only visual"):
        visual.load_clip()
    assert stub_open_clip == []


@LOADS
def test_load_with_weights_present_uses_the_file(stub_open_clip, no_network, load):
    weights = _fake_weights()
    with pytest.raises(_Loaded):
        load()
    assert stub_open_clip == [((visual.CLIP_ARCH,), {"pretrained": str(weights)})]


@LOADS
def test_missing_weights_raise_a_clear_error_without_loading(stub_open_clip, no_network, load):
    with pytest.raises(visual.ClipWeightsMissing) as err:
        load()
    assert str(err.value) == (
        f"CLIP weights not found in {paths.clip_dir()}; run: cutpoint setup --only visual"
    )
    assert isinstance(err.value, FileNotFoundError)
    assert stub_open_clip == []  # open_clip (and with it the Hub) was never asked


def test_real_open_clip_loads_the_local_file_without_the_hub(monkeypatch, no_network):
    """The real open_clip factory routes ``pretrained=<file>`` straight to the checkpoint
    loader: no tag lookup, no ``download_pretrained``. The model class and the checkpoint
    loader are fakes, so no weights are read."""
    open_clip = pytest.importorskip("open_clip")
    torch = pytest.importorskip("torch")
    from open_clip import factory  # noqa: PLC0415

    weights = _fake_weights()
    built: list[dict[str, Any]] = []
    loaded: list[str] = []

    class FakeClip(torch.nn.Module):
        def __init__(self, **cfg: Any) -> None:
            super().__init__()
            built.append(cfg)
            self.visual = torch.nn.Module()
            self.visual.image_size = 224

    monkeypatch.setattr(factory, "CLIP", FakeClip)
    monkeypatch.setattr(factory, "download_pretrained", _no_network)
    monkeypatch.setattr(factory, "load_checkpoint", lambda m, path, **kw: loaded.append(path))
    model = visual.load_clip()
    assert isinstance(model, FakeClip)
    assert loaded == [str(weights)]
    assert built[0]["quick_gelu"] is True  # the arch's own QuickGELU, as with the tag
    assert open_clip.create_model_and_transforms is factory.create_model_and_transforms


def test_setup_loads_by_tag_so_the_hub_downloads(stub_open_clip):
    with pytest.raises(_Loaded):
        visual.load_clip(download=True)
    assert stub_open_clip == [
        (
            (visual.CLIP_ARCH,),
            {"pretrained": visual.CLIP_PRETRAINED, "cache_dir": str(paths.clip_dir())},
        )
    ]


def test_install_visual_is_the_downloading_load(monkeypatch):
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(setup, "_download_mirrors", lambda *a, **kw: None)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        visual,
        "VisualModels",
        lambda **kw: seen.append(kw) or types.SimpleNamespace(close=lambda: None),
    )
    setup.install_visual()
    assert seen == [{"download": True}]


def test_ski_layer_features_skip_visual_without_clip_weights(monkeypatch, caplog, tmp_path):
    """Cached visual data with captions that have no cached embeddings, and no weights:
    the visual features are left out with a WARNING and the ranking goes on."""
    from cutpoint.events import voice  # noqa: PLC0415
    from cutpoint.pano import views  # noqa: PLC0415

    def no_weights(*_a: Any, **_kw: Any) -> Any:
        raise visual.ClipWeightsMissing("CLIP weights not found in x; run: cutpoint setup")

    monkeypatch.setattr(voice, "load_voice", lambda src: None)
    monkeypatch.setattr(views, "layer_grids", lambda *a: None)
    monkeypatch.setattr(visual, "load_visual", lambda src: types.SimpleNamespace(t=np.arange(3.0)))
    monkeypatch.setattr(visual, "frame_scores", no_weights)
    out = actioncam.layer_features(tmp_path / "a.mp4", 10)
    assert out == {}
    assert messages(caplog) == [
        "a.mp4: visual features skipped: CLIP weights not found in x; run: cutpoint setup"
    ]
