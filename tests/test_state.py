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
    features,
    ffmpeg_utils,
    refmatch,
    reset_state,
    stage,
    state,
    styles,
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


def _fill_encoder(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    def fail(*a, **k):
        raise FFmpegError("no GPU encoder")

    mp.setattr(ffmpeg_utils, "run", fail)
    ffmpeg_utils.video_encoder()  # fills encoder_error (the probes) and video_encoder


def _fill_decoder(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    def fail(*a, **k):
        raise FFmpegError("no GPU decoder")

    mp.setattr(ffmpeg_utils, "run", fail)
    # a candidate whatever GPU (or none, on CI) this machine has, so the probe cache is filled
    mp.setattr(ffmpeg_utils, "decoder_candidates", lambda: [ffmpeg_utils.HwDecoder("cuda")])
    ffmpeg_utils.gpu_decoder()  # fills decoder_error (the probes) and gpu_decoder


def _fill_soxr(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    mp.setattr(ffmpeg_utils, "run", lambda *a, **k: "")
    ffmpeg_utils.has_soxr()


def _fill_filters(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    mp.setattr(ffmpeg_utils, "run", lambda *a, **k: "")
    ffmpeg_utils.ffmpeg_filters()


def _fill_aesthetic(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    from cutpoint.events import visual  # noqa: PLC0415

    mp.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: {}))
    visual._aesthetic_layers(write(tmp / "head.pth"))


def _fill_torch_home(tmp: Path, mp: pytest.MonkeyPatch) -> None:
    from cutpoint.events import paths as epaths  # noqa: PLC0415

    mp.delenv("TORCH_HOME", raising=False)
    epaths.apply_torch_home()


# conftest stubs config._drive_remote per test; keep the real cached function to fill.
_REAL_DRIVE_REMOTE = config._drive_remote

# How to put one item into each registered lru/functools cache.
CACHE_FILLERS: dict[str, Callable[[Path, pytest.MonkeyPatch], None]] = {
    "cutpoint.config.get_config": lambda t, m: config.get_config(),
    "cutpoint.config._prefix_forms": lambda t, m: config._prefix_forms(str(t)),
    "cutpoint.config._drive_remote": lambda t, m: _REAL_DRIVE_REMOTE("Z:"),
    "cutpoint.stage._stage_index": lambda t, m: stage._stage_index(),
    "cutpoint.stage.staged_path": lambda t, m: stage.staged_path("-"),
    "cutpoint.ffmpeg_utils.find_tool": _fill_find_tool,
    "cutpoint.ffmpeg_utils.encoder_error": _fill_encoder,
    "cutpoint.ffmpeg_utils.decoder_error": _fill_decoder,
    "cutpoint.ffmpeg_utils.gpu_decoder": _fill_decoder,
    "cutpoint.ffmpeg_utils.has_soxr": _fill_soxr,
    "cutpoint.ffmpeg_utils.ffmpeg_filters": _fill_filters,
    "cutpoint.events.visual._aesthetic_layers": _fill_aesthetic,
    "cutpoint.events.paths._TORCH_HOME_SET": _fill_torch_home,
    "cutpoint.features.installed_extras": lambda t, m: features.installed_extras(),
    "cutpoint.styles._warn_unknown": lambda t, m: styles._warn_unknown("x.json", ("x",)),
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
    names = set(state._REGISTRY)
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


@pytest.mark.parametrize("name", sorted(state._REGISTRY))
def test_reset_state_clears_entry(name, tmp_path, monkeypatch):
    entry = state._REGISTRY[name]
    _fill(name, tmp_path, monkeypatch)
    assert entry.size() > 0
    reset_state(models=True)
    assert entry.size() == 0


def test_reset_state_keeps_models_by_default(tmp_path, monkeypatch):
    models = [e for e in state._REGISTRY.values() if e.model]
    assert "cutpoint.events.visual._TEXT_MODEL" in [e.name for e in models]
    for entry in models:
        _fill(entry.name, tmp_path, monkeypatch)
    reset_state()
    assert all(e.size() > 0 for e in models)
    reset_state(models=True)
    assert all(e.size() == 0 for e in models)


def test_register_dict_cache_gives_cache_api_and_is_reset(monkeypatch):
    monkeypatch.setattr(state, "_REGISTRY", dict(state._REGISTRY))  # keep the real one clean
    store = {"a": 1, "b": 2}

    def fn():
        return store

    assert state.register_dict_cache(fn, store) is fn
    assert fn.cache_info().currsize == 2
    reset_state()
    assert store == {}
    assert fn.cache_info().currsize == 0
    store["c"] = 3
    fn.cache_clear()
    assert store == {}


def test_config_edit_picked_up_without_reset(home, tmp_path):
    """#137: an edited config.json is re-read; a broken one raises every time, not cached."""
    cfg = home / "config.json"

    def edit(text: str, step: int) -> None:
        write(cfg, text)
        os.utime(cfg, ns=(step * 10**9, step * 10**9))  # distinct stamp, whatever the size

    edit(json.dumps({"projects_dir": str(tmp_path / "a")}), 1)
    assert config.projects_dir() == tmp_path / "a"
    edit(json.dumps({"projects_dir": str(tmp_path / "b")}), 2)
    assert config.projects_dir() == tmp_path / "b"
    edit("{broken", 3)
    for _ in range(2):
        with pytest.raises(ValueError):
            config.projects_dir()
    edit(json.dumps({"projects_dir": str(tmp_path / "c")}), 4)
    assert config.projects_dir() == tmp_path / "c"


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


def test_ollama_url_bad_port_raises_but_unload_does_not(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "host:abc")
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
    assert vision.VisionRater().url == "http://gpu-box:1234"


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
def _non_json() -> object:
    raise ValueError("Expecting value")


@pytest.mark.parametrize(
    "body",
    [_non_json, lambda: ["llava"]],
    ids=["non-json", "not-an-object"],
)
def test_vision_unavailable_when_ollama_reply_is_not_a_json_object(monkeypatch, body):
    resp = SimpleNamespace(raise_for_status=lambda: None, json=body)
    monkeypatch.setattr(vision.requests, "get", lambda *_a, **_k: resp)
    assert vision.VisionRater(url="http://x:1", model="m").available() is False


@pytest.mark.parametrize("module", ["voice", "visual"])
def test_offline_footage_has_no_voice_or_visual_data(home, tmp_path, caplog, module):
    layer = importlib.import_module(f"cutpoint.events.{module}")
    load = getattr(layer, f"load_{module}")
    with caplog.at_level("WARNING"):
        assert load(tmp_path / "gone.mp4") is None
    assert caplog.text.count("not reachable") == 1
