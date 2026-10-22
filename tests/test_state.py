"""Tests for cutpoint.state / cutpoint.reset_state and call-time settings."""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import cutpoint
from conftest import write
from cutpoint import (
    backends,
    cache,
    config,
    ffmpeg_utils,
    refmatch,
    reset_state,
    stage,
    state,
    vision,
)
from cutpoint.events import common
from cutpoint.ffmpeg_utils import FFmpegError

# Import every module so the registry is complete (``__main__`` modules run the CLI).
for _mod in pkgutil.walk_packages(cutpoint.__path__, "cutpoint."):
    if not _mod.name.endswith("__main__"):
        importlib.import_module(_mod.name)


def _fill_find_tool(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    mp.setenv("CUTPOINT_FFMPEG", str(write(tmp / "ffmpeg.exe")))
    ffmpeg_utils.find_tool("ffmpeg")


def _fill_nvenc(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    def fail(*a, **k):
        raise FFmpegError("no nvenc")

    mp.setattr(ffmpeg_utils, "run", fail)
    ffmpeg_utils.nvenc_available()


def _fill_aesthetic(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    from cutpoint.events import visual  # noqa: PLC0415

    mp.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: {}))
    visual._aesthetic_layers(write(tmp / "head.pth"))


def _fill_torch_home(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    from cutpoint.events import paths as epaths  # noqa: PLC0415

    mp.delenv("TORCH_HOME", raising=False)
    epaths.apply_torch_home()


# How to put one item into each registered lru/functools cache.
CACHE_FILLERS: dict[str, Callable[[Path, pytest.MonkeyPatch], None]] = {
    "cutpoint.config.get_config": lambda t, m: config.get_config(),
    "cutpoint.config._prefix_forms": lambda t, m: config._prefix_forms(str(t)),
    "cutpoint.config._drive_remote": lambda t, m: config._drive_remote("Z:"),
    "cutpoint.stage._stage_index": lambda t, m: stage._stage_index(),
    "cutpoint.stage.staged_path": lambda t, m: stage.staged_path("-"),
    "cutpoint.ffmpeg_utils.find_tool": _fill_find_tool,
    "cutpoint.ffmpeg_utils.nvenc_available": _fill_nvenc,
    "cutpoint.events.visual._aesthetic_layers": _fill_aesthetic,
    "cutpoint.events.paths._TORCH_HOME_SET": _fill_torch_home,
}


def _fill(name: str, tmp: Path, mp: pytest.MonkeyPatch) -> None:
    if name in CACHE_FILLERS:
        CACHE_FILLERS[name](tmp, mp)
        return
    module, attr = name.rsplit(".", 1)
    obj = getattr(importlib.import_module(module), attr)
    if isinstance(obj, set):
        obj.add("x")
    else:
        obj["x"] = 1


def test_every_module_cache_is_registered():
    names = set(state.registered())
    unregistered = [
        f"{mod.__name__}.{obj.__qualname__}"
        for mod in list(sys.modules.values())
        if mod is not None and mod.__name__.startswith("cutpoint")
        for obj in list(vars(mod).values())
        if hasattr(obj, "cache_clear")
        and not isinstance(obj, type)  # e.g. the state._Cached protocol
        and getattr(obj, "__module__", None) == mod.__name__
    ]
    assert unregistered
    assert [n for n in unregistered if n not in names] == []


@pytest.mark.parametrize("name", sorted(state.registered()))
def test_reset_state_clears_entry(name, tmp_path, monkeypatch):
    entry = state.registered()[name]
    _fill(name, tmp_path, monkeypatch)
    assert entry.size() > 0
    reset_state(models=True)
    assert entry.size() == 0


def test_reset_state_keeps_models_by_default(tmp_path, monkeypatch):
    models = [e for e in state.registered().values() if e.model]
    assert [e.name for e in models] == ["cutpoint.events.visual._TEXT_MODEL"]
    _fill(models[0].name, tmp_path, monkeypatch)
    reset_state()
    assert models[0].size() == 1
    reset_state(models=True)
    assert models[0].size() == 0


def test_projects_dir_change_picked_up_after_reset(home, tmp_path):
    write(home / "config.json", json.dumps({"projects_dir": str(tmp_path / "a")}))
    assert config.projects_dir() == tmp_path / "a"
    write(home / "config.json", json.dumps({"projects_dir": str(tmp_path / "b")}))
    assert config.projects_dir() == tmp_path / "a"  # cached for the process
    reset_state()
    assert config.projects_dir() == tmp_path / "b"


def test_cutpoint_home_change_picked_up_after_reset(home, tmp_path, monkeypatch):
    assert config.projects_dir() == home / "projects"
    monkeypatch.setenv("CUTPOINT_HOME", str(tmp_path / "other"))
    reset_state()
    assert config.projects_dir() == tmp_path / "other" / "projects"


# -- Ollama URL -----------------------------------------------------------------------
def test_ollama_url_default_config_and_env(home, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert backends.ollama_url() == "http://127.0.0.1:11434"
    write(home / "config.json", json.dumps({"ollama_url": "http://cfg:1/"}))
    reset_state()
    assert backends.ollama_url() == "http://cfg:1"
    monkeypatch.setenv("OLLAMA_HOST", "remote:11434")  # read at call time: no reset needed
    assert backends.ollama_url() == "http://remote:11434"


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("myhost", "http://myhost:11434"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("https://h/", "https://h"),
    ],
)
def test_ollama_url_normalised(host, url, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert backends.ollama_url() == url


@pytest.mark.parametrize("host", ["host:abc"])
def test_ollama_url_bad_port_raises_but_unload_does_not(host, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", host)
    with pytest.raises(ValueError, match="invalid port"):
        backends.ollama_url()
    monkeypatch.setattr(common.requests, "get", lambda *a, **k: pytest.fail("no request"))
    assert common.unload_ollama() == []


def test_vision_and_unload_use_the_same_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:1234")
    seen: list[str] = []

    def fake_get(url, **k):
        seen.append(url)
        raise requests.ConnectionError

    monkeypatch.setattr(common.requests, "get", fake_get)
    assert common.unload_ollama() == []
    assert seen == ["http://gpu-box:1234/api/ps"]
    assert vision.default_url() == "http://gpu-box:1234"


# -- F11: refmatch audio failures keyed by cache folder ------------------------------
def test_audio_failure_keyed_by_source_cache_dir(tmp_path, monkeypatch):
    calls: list[Path] = []

    def fail(path, sr):
        calls.append(path)
        raise FFmpegError("decode failed")

    monkeypatch.setattr(refmatch, "decode_pcm", fail)
    monkeypatch.setattr(refmatch, "probe", lambda p: SimpleNamespace(audio_streams=1))
    src = write(tmp_path / "clip.mp4", "a")
    assert refmatch.audio_8k(src).size == 0
    assert refmatch.audio_8k(src).size == 0
    assert len(calls) == 1
    failed = refmatch._AUDIO_FAILED
    assert failed == {str(cache.source_dir(src))}
    write(src, "changed")  # new size -> new cache folder -> retried
    refmatch.audio_8k(src)
    assert len(calls) == 2
    reset_state()
    assert not refmatch._AUDIO_FAILED
    refmatch.audio_8k(src)
    assert len(calls) == 3


# -- TORCH_HOME -----------------------------------------------------------------------
def test_torch_home_follows_cutpoint_home_but_not_the_user(home, tmp_path, monkeypatch):
    from cutpoint.events import paths as epaths  # noqa: PLC0415

    monkeypatch.delenv("TORCH_HOME", raising=False)
    epaths.apply_torch_home()
    assert os.environ["TORCH_HOME"] == str(home / "models" / "torch")
    monkeypatch.setenv("CUTPOINT_HOME", str(tmp_path / "other"))
    epaths.apply_torch_home()  # still ours: follows the new home
    assert os.environ["TORCH_HOME"] == str(tmp_path / "other" / "models" / "torch")
    reset_state()  # undoes our value
    assert "TORCH_HOME" not in os.environ
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "mine"))
    epaths.apply_torch_home()  # the user's own value is kept
    assert os.environ["TORCH_HOME"] == str(tmp_path / "mine")


# -- Vision / visual cache: unreachable inputs degrade quietly --------------------------
def test_vision_unavailable_when_ollama_returns_non_json(monkeypatch):
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint import vision  # noqa: PLC0415

    def _html(*_a, **_k):
        def _bad_json():
            raise ValueError("Expecting value")

        return SimpleNamespace(raise_for_status=lambda: None, json=_bad_json)

    monkeypatch.setattr(vision.requests, "get", _html)
    assert vision.VisionRater(url="http://x:1", model="m").available() is False


@pytest.mark.parametrize("payload", [["llava"], "llava", 3])
def test_vision_unavailable_when_ollama_json_is_not_an_object(monkeypatch, payload):
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint import vision  # noqa: PLC0415

    resp = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(vision.requests, "get", lambda *_a, **_k: resp)
    assert vision.VisionRater(url="http://x:1", model="m").available() is False


def test_offline_footage_has_no_voice_data(home, tmp_path, caplog):
    from cutpoint.events import voice  # noqa: PLC0415

    with caplog.at_level("WARNING"):
        assert voice.load_voice(tmp_path / "gone.mp4") is None
    assert caplog.text.count("not reachable") == 1


def test_load_visual_offline_footage_is_none_with_warning(home, tmp_path, caplog):
    from cutpoint.events import visual  # noqa: PLC0415

    with caplog.at_level("WARNING"):
        assert visual.load_visual(tmp_path / "gone.mp4") is None
    assert "not reachable" in caplog.text
