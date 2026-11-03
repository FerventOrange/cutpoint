"""The Faster R-CNN detector loads from the local checkpoint only outside ``cutpoint setup``
(#654): no ``load_state_dict_from_url``, no download, a clear error when the file is missing."""

from __future__ import annotations

import socket
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from conftest import write
from cutpoint import cli
from cutpoint.events import paths, setup, visual
from cutpoint.pano import qa, views
from cutpoint.pano.x3 import X3Clip


class _Network(AssertionError):
    """Raised by every patched network entry point."""


def _no_network(*_a: Any, **_kw: Any) -> Any:
    raise _Network("network call during a detector load")


@pytest.fixture
def no_network(monkeypatch):
    """Every way a detector load could reach the network raises :class:`_Network`."""
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    try:
        import torch.hub  # noqa: PLC0415

        monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _no_network)
        monkeypatch.setattr(torch.hub, "download_url_to_file", _no_network)
    except ImportError:
        pass
    try:
        from torchvision.models import _api  # noqa: PLC0415

        monkeypatch.setattr(_api, "load_state_dict_from_url", _no_network)
    except ImportError:
        pass


def _fake_ckpt() -> Path:
    """A fake checkpoint where ``cutpoint setup`` leaves the detector (torch hub cache)."""
    return write(paths.detector_weights(), b"fake")


class _Det:
    """The stub detector model: records the state dict it gets."""

    def __init__(self, rec: types.SimpleNamespace, kw: dict[str, Any]) -> None:
        rec.built.append(kw)
        self.rec = rec

    def load_state_dict(self, sd: Any) -> None:
        self.rec.state.append(sd)

    def eval(self) -> _Det:
        return self

    def to(self, *_a: Any) -> _Det:
        return self


@pytest.fixture
def stub_torch(monkeypatch):
    """Stub torch and torchvision; the detector builder and ``torch.load`` record their calls.
    CLIP is a fake that loads nothing."""
    rec = types.SimpleNamespace(built=[], state=[], loaded=[])

    def load(f, **kw):
        rec.loaded.append((f, kw))
        return {"w": 1}

    detection = types.SimpleNamespace(
        fasterrcnn_resnet50_fpn_v2=lambda **kw: _Det(rec, kw),
        FasterRCNN_ResNet50_FPN_V2_Weights=types.SimpleNamespace(COCO_V1="COCO_V1"),
    )
    torchvision = types.SimpleNamespace(models=types.SimpleNamespace(detection=detection))
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float16="f16",
        float32="f32",
        load=load,
        tensor=lambda *a, **kw: types.SimpleNamespace(view=lambda *s: None),
    )
    clip = types.SimpleNamespace(eval=lambda: clip, to=lambda *a: clip)
    monkeypatch.setitem(sys.modules, "open_clip", types.SimpleNamespace(get_tokenizer=str))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setattr(visual, "load_clip", lambda download=False: clip)
    return rec


SIZE = {"box_score_thresh": 0.3, "min_size": 360, "max_size": 641}


def test_present_checkpoint_loads_from_the_file(no_network, stub_torch):
    ckpt = _fake_ckpt()
    visual.VisualModels()
    assert stub_torch.built == [
        {"weights": None, "weights_backbone": None, "num_classes": 91, **SIZE}
    ]
    assert stub_torch.loaded == [(ckpt, {"map_location": "cpu", "weights_only": True})]
    assert stub_torch.state == [{"w": 1}]


def test_missing_checkpoint_raises_a_clear_error_without_building(no_network, stub_torch):
    with pytest.raises(visual.DetectorWeightsMissing) as err:
        visual.VisualModels()
    ckpt = paths.detector_weights()
    assert str(err.value) == (
        f"Faster R-CNN detector weights not found in {ckpt.parent}; "
        "run: cutpoint setup --only visual"
    )
    assert isinstance(err.value, visual.WeightsMissing)
    assert isinstance(err.value, FileNotFoundError)
    assert stub_torch.built == [] and stub_torch.loaded == []


def test_clip_error_shares_the_base_class():
    assert issubclass(visual.ClipWeightsMissing, visual.WeightsMissing)


def test_setup_load_asks_torchvision_for_the_coco_weights(stub_torch):
    visual.VisualModels(download=True)
    assert stub_torch.built == [{"weights": "COCO_V1", **SIZE}]
    assert stub_torch.loaded == []  # torchvision fetches and loads them itself


def test_install_visual_still_downloads_the_detector(monkeypatch, stub_torch):
    monkeypatch.setattr(setup, "_download_mirrors", lambda *a, **kw: None)
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(visual.VisualModels, "close", lambda self: None)
    setup.install_visual()
    assert [kw["weights"] for kw in stub_torch.built] == ["COCO_V1"]


def test_real_torchvision_builds_without_any_download(monkeypatch, no_network):
    """The real builder with ``weights=None, weights_backbone=None`` fetches nothing (neither
    the COCO nor the ImageNet backbone weights); the checkpoint read is a fake."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    ckpt = _fake_ckpt()
    got: list[Any] = []
    monkeypatch.setattr(torch, "load", lambda f, **kw: got.append(f) or {"sd": 1})
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", lambda self, sd: got.append(sd))
    model = visual.load_detector(0.3, 360, 641)
    assert type(model).__name__ == "FasterRCNN"
    assert model.roi_heads.box_predictor.cls_score.out_features == 91
    assert got == [ckpt, {"sd": 1}]


# --------------------------------------------------------------------------- 360 commands
def test_pano_qa_fails_with_the_error_and_no_folder(no_network, stub_torch, tmp_path, monkeypatch):
    d = tmp_path / "preview"
    d.mkdir()
    monkeypatch.setattr(qa, "find_inputs", lambda qa_dir: [{"file": "001_a.mp4"}])
    with pytest.raises(visual.DetectorWeightsMissing):
        qa.run_qa(d, json_only=True)
    assert not (tmp_path / "preview_qa").exists()
    assert stub_torch.built == []


def test_pano_qa_cli_reports_not_found(no_network, stub_torch, tmp_path, monkeypatch, capsys):
    from conftest import envelope  # noqa: PLC0415

    d = tmp_path / "preview"
    d.mkdir()
    monkeypatch.setattr(qa, "find_inputs", lambda qa_dir: [{"file": "001_a.mp4"}])
    assert cli.main(["360", "qa", str(d), "--json-only", "--json"]) == 1
    doc = envelope(capsys.readouterr().out)
    assert doc["error"]["code"] == "not_found"
    assert doc["error"]["message"].startswith("Faster R-CNN detector weights not found in ")


def test_pano_views_layer_loads_locally(no_network, stub_torch, tmp_path, monkeypatch):
    lrv = write(tmp_path / "VID_001.lrv", b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=10, duration=10.0)
    monkeypatch.setattr(views, "load_views", lambda src, p=None: None)
    monkeypatch.setattr(visual, "text_embeddings", lambda prompts, models=None: None)
    monkeypatch.setattr(visual.VisualModels, "close", lambda self: None)
    monkeypatch.setattr("cutpoint.insta360.imu_source", lambda src: None)  # skip the clip
    with pytest.raises(visual.DetectorWeightsMissing):
        views.run_layer([clip])
    assert stub_torch.built == []
    ckpt = _fake_ckpt()
    views.run_layer([clip])
    assert stub_torch.built == [
        {
            "weights": None,
            "weights_backbone": None,
            "num_classes": 91,
            "box_score_thresh": views.ViewParams().det_thresh,
            "min_size": 512,
            "max_size": 911,
        }
    ]
    assert stub_torch.loaded[0][0] == ckpt
