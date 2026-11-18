"""A shared stand-in for the clip encoder ``export_clip_checked`` (no ffmpeg).

``fake_export`` is an opt-in fixture (registered in conftest's ``pytest_plugins``);
``install_fake_export`` does the same from a helper or mid-test. Every call is bound to the real
signature, so a fake call the real encoder would reject fails the test."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cutpoint import clips, encode, extend

EXPORT_SIGNATURE = inspect.signature(encode.export_clip_checked)


class FakeExport:
    """``export_clip_checked`` that writes ``content`` to ``out`` and returns None (the clip is
    complete). ``on_call`` (if set) is given the bound arguments of each call."""

    def __init__(self, content: bytes = b"new") -> None:
        self.content = content
        self.on_call: Callable[[inspect.BoundArguments], Any] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        bound = EXPORT_SIGNATURE.bind(*args, **kwargs)
        if self.on_call is not None:
            self.on_call(bound)
        Path(bound.arguments["out"]).write_bytes(self.content)


def install_fake_export(
    monkeypatch: pytest.MonkeyPatch, *modules: ModuleType, **kw: Any
) -> FakeExport:
    """Replace ``export_clip_checked`` in each of ``modules`` (``clips``, ``extend``) with one
    :class:`FakeExport` (``kw``: its ``content``) and return it."""
    fake = FakeExport(**kw)
    for module in modules:
        monkeypatch.setattr(module, "export_clip_checked", fake)
    return fake


@pytest.fixture
def fake_export(monkeypatch: pytest.MonkeyPatch) -> FakeExport:
    """The encoder of ``clips`` and ``extend`` faked: each clip is the bytes ``b"new"`` (set
    ``.content`` to change it); probe and the sources are left alone."""
    return install_fake_export(monkeypatch, clips, extend)
