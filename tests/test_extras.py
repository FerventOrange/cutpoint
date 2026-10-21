"""Tests for optional-dependency messages (extras.py) and how cli.main reports them."""

from __future__ import annotations

import argparse
import logging

import pytest

from cutpoint import cli, extras


def test_missing_module_message():
    msg = extras.missing_module_message("telemetry_parser")
    assert msg is not None and msg.startswith("telemetry_parser is not installed")
    assert "'ski' extra" in msg
    msg = extras.missing_module_message("torch.nn.functional")  # top-level package decides
    assert msg is not None and msg.startswith("torch is not installed")
    assert "'events' extra" in msg
    assert extras.missing_module_message("definitely_not_an_extra") is None


def test_install_hint_uv_and_pip(monkeypatch):
    monkeypatch.setenv("UV", "1")
    assert extras.install_hint("events").startswith("uv sync --extra events")
    monkeypatch.delenv("UV")
    monkeypatch.setattr(extras.importlib.util, "find_spec", lambda name: object())
    assert "-m pip install" in extras.install_hint("events")


def _patch_paths_command(monkeypatch, exc: BaseException) -> None:
    def handler(args: argparse.Namespace) -> int:
        raise exc

    monkeypatch.setattr(cli, "_cmd_paths", handler)


def test_main_returns_1_for_missing_known_optional_module(monkeypatch, caplog):
    _patch_paths_command(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    with caplog.at_level(logging.ERROR):
        assert cli.main(["paths"]) == 1
    assert any(
        "torch is not installed" in r.getMessage() and "'events' extra" in r.getMessage()
        for r in caplog.records
    )


def test_main_reraises_for_unknown_missing_module(monkeypatch):
    _patch_paths_command(
        monkeypatch, ModuleNotFoundError("No module named 'bogus_pkg'", name="bogus_pkg")
    )
    with pytest.raises(ModuleNotFoundError, match="bogus_pkg"):
        cli.main(["paths"])
