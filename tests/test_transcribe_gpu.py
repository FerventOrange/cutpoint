"""GPU choice for whisper: CUDA only when a device exists and its libraries load."""

import ctypes
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
