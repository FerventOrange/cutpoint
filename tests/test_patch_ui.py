"""The `patch_ui` helper (conftest): one patch reaches every `cutpoint.ui` module holding the name."""

from __future__ import annotations

import sys
import types

import pytest

from conftest import patch_ui
from cutpoint import ui


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def test_patch_ui_reaches_every_ui_module_holding_the_same_object(monkeypatch):
    name = "Jobs"  # any attribute of the facade
    old = getattr(ui, name)
    sub = _fake_module("cutpoint.ui.fakesub", **{name: old})
    other = _fake_module("cutpoint.ui.fakeother", **{name: object()})
    sibling = _fake_module("cutpoint.uix", **{name: old})  # not a cutpoint.ui module
    for mod in (sub, other, sibling):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    sentinel = object()
    with pytest.MonkeyPatch.context() as inner:
        patch_ui(inner, name, sentinel)
        assert getattr(ui, name) is sentinel
        assert getattr(sub, name) is sentinel
        assert getattr(other, name) is not sentinel  # another object under the same name
        assert getattr(sibling, name) is old  # `cutpoint.uix` is not `cutpoint.ui.*`
    assert getattr(ui, name) is old  # undone with the monkeypatch
    assert getattr(sub, name) is old


def test_patch_ui_refuses_a_name_the_ui_does_not_have(monkeypatch):
    with pytest.raises(AssertionError, match="no_such_name"):
        patch_ui(monkeypatch, "no_such_name", 1)


def test_patch_ui_with_a_none_value_skips_modules_lacking_the_name(monkeypatch):
    monkeypatch.setattr(ui, "_none_attr", None, raising=False)
    bare = _fake_module("cutpoint.ui.fakebare")  # has no such attribute
    monkeypatch.setitem(sys.modules, bare.__name__, bare)
    sentinel = object()
    with pytest.MonkeyPatch.context() as inner:
        patch_ui(inner, "_none_attr", sentinel)
        assert ui._none_attr is sentinel
        assert not hasattr(bare, "_none_attr")
    assert ui._none_attr is None
