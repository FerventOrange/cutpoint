"""huggingface_hub's duplicate stderr handler is dropped before every model load."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from cutpoint import hub
from cutpoint.crash import ClipEncoder
from cutpoint.events import visual


class _Loaded(Exception):
    """Raised by the stub open_clip: the model load was reached."""


@pytest.fixture
def hub_handler(monkeypatch):
    """A handler on the Hub's logger, as huggingface_hub adds on import; yields the logger."""
    pytest.importorskip("huggingface_hub.utils.logging")
    hf_logger = logging.getLogger("huggingface_hub")
    monkeypatch.setattr(hf_logger, "handlers", [logging.StreamHandler()])
    monkeypatch.setattr(hub, "_hub_handler_removed", False)
    return hf_logger


@pytest.fixture
def stub_open_clip(monkeypatch, hub_handler):
    """Stub open_clip, torch and torchvision; the open_clip load records the Hub's handlers."""
    seen: list[list[logging.Handler]] = []

    def create_model_and_transforms(*_a, **_kw):
        seen.append(list(hub_handler.handlers))
        raise _Loaded

    open_clip = types.SimpleNamespace(create_model_and_transforms=create_model_and_transforms)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False), float16="f16", float32="f32"
    )
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace())
    return seen


def test_quiet_hub_handler_removes_the_hub_handler(hub_handler):
    hub.quiet_hub_handler()
    assert hub_handler.handlers == []


@pytest.mark.parametrize(
    "load",
    [
        lambda: ClipEncoder(),
        lambda: visual.VisualModels(),
        lambda: visual.text_embeddings(("a dog",)),
    ],
    ids=["ClipEncoder", "VisualModels", "text_embeddings"],
)
def test_clip_load_drops_the_hub_handler_first(stub_open_clip, hub_handler, load):
    with pytest.raises(_Loaded):
        load()
    assert stub_open_clip == [[]]
    assert hub_handler.handlers == []
