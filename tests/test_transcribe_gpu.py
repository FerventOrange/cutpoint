"""GPU choice for whisper: CUDA only when a device exists and its libraries load."""

import ctypes
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import types

import pytest

from cutpoint import transcribe


@pytest.fixture
def nvidia_device(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    )
    monkeypatch.setattr(transcribe, "_cpu_notice_logged", False)
    monkeypatch.setattr(transcribe, "_torch_loaded", True)  # never import a real torch here


def set_libs_loadable(monkeypatch, loadable: bool) -> None:
    def load(name, *args, **kwargs):
        if not loadable:
            raise OSError(name)

    # WinDLL only exists on Windows; the code calls whichever matches the platform.
    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    monkeypatch.setattr(ctypes, "CDLL", load)


def hints(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == transcribe.logger.name]


def test_missing_cuda_libs_fall_back_to_cpu_and_hint_once(nvidia_device, monkeypatch, caplog):
    set_libs_loadable(monkeypatch, False)
    with caplog.at_level(logging.INFO, logger=transcribe.logger.name):
        assert [transcribe.cuda_available() for _ in range(3)] == [False] * 3
    (rec,) = hints(caplog)
    assert rec.levelno == logging.INFO
    assert "'gpu' extra" in rec.getMessage()


def test_loadable_cuda_libs_choose_cuda_silently(nvidia_device, monkeypatch, caplog):
    set_libs_loadable(monkeypatch, True)
    with caplog.at_level(logging.INFO, logger=transcribe.logger.name):
        assert transcribe.cuda_available() is True
    assert hints(caplog) == []


class FakeTorchFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves a fake ``torch`` through the real import machinery and records the import."""

    def __init__(self, calls: list[str], error: Exception | None = None) -> None:
        self.calls, self.error = calls, error

    def find_spec(self, name, path=None, target=None):
        return importlib.machinery.ModuleSpec(name, self) if name == "torch" else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.calls.append("torch")
        if self.error:
            raise self.error


@pytest.fixture
def import_order(monkeypatch):
    """Calls in order: 'torch' (imported), 'ctranslate2' (CUDA probe)."""
    calls: list[str] = []

    def device_count():
        calls.append("ctranslate2")
        return 1

    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=device_count)
    )
    monkeypatch.setitem(sys.modules, "torch", None)  # restored to absent afterwards
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(transcribe, "_torch_loaded", False)
    monkeypatch.setattr(transcribe, "_cpu_notice_logged", False)
    set_libs_loadable(monkeypatch, True)
    return calls


def install_torch(monkeypatch, calls: list[str], error: Exception | None = None) -> None:
    monkeypatch.setattr(sys, "meta_path", [FakeTorchFinder(calls, error), *sys.meta_path])
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: object() if name == "torch" else real_find_spec(name, *a),
    )


def test_torch_is_imported_before_the_ctranslate2_probe_once(import_order, monkeypatch):
    install_torch(monkeypatch, import_order)
    assert transcribe.cuda_available() is True
    assert transcribe.cuda_available() is True
    assert import_order == ["torch", "ctranslate2", "ctranslate2"]


def test_failing_torch_import_is_ignored(import_order, monkeypatch, caplog):
    install_torch(monkeypatch, import_order, OSError("[WinError 127] cudnn_cnn64_9.dll"))
    with caplog.at_level(logging.DEBUG, logger=transcribe.logger.name):
        assert transcribe.cuda_available() is True
    assert import_order == ["torch", "ctranslate2"]
    assert any("WinError 127" in r.getMessage() for r in hints(caplog))


def test_missing_torch_is_not_imported(import_order, monkeypatch):
    finder = FakeTorchFinder(import_order)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    assert transcribe.cuda_available() is True
    assert import_order == ["ctranslate2"]
