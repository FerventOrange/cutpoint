"""Pre-flight checks: model files per layer, fail-fast ordering, Ollama unload, 360 analyze."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pytest

from cutpoint import activity
from cutpoint import analyze as analyze_mod
from cutpoint.events import common, paths, pipeline
from cutpoint.pano import cli, views


def _create(items: list[tuple[str, Path]]) -> None:
    for _, p in items:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")


def test_clip_weights_found_in_hf_snapshot(home: Path):
    assert not paths.clip_weights().is_file()
    repo = paths.clip_dir() / "models--timm--vit_large_patch14_clip_224.openai"
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    f = snap / "open_clip_model.safetensors"
    f.write_text("x", encoding="utf-8")
    assert paths.clip_weights() == f


# --------------------------------------------------------------------------- pipeline
@pytest.fixture
def extra_ok(monkeypatch):
    """The events extra counts as installed; no llama-server on PATH."""
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(paths.shutil, "which", lambda *a, **k: None)


def test_preflight_lists_everything_missing_in_setup_order(home: Path, extra_ok):
    layers = ("voice", "vlm", "visual")
    with pytest.raises(FileNotFoundError) as exc:
        pipeline.preflight(layers)
    msg = str(exc.value)
    for _, p in paths.missing_files(layers):
        assert str(p) in msg
    # setup order is pip, panns, llama, omni, visual, voice
    assert msg.endswith("cutpoint setup --only panns llama omni visual voice")


def test_preflight_passes_when_files_present_or_ocr(home: Path, extra_ok):
    pipeline.preflight(("ocr",))
    _create(paths.layer_files(["audio"]))
    pipeline.preflight(("audio",))


def test_run_events_missing_models_fails_before_unload(
    home: Path, monkeypatch, tmp_path: Path, extra_ok
):
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: {"sources": []})
    calls = []
    monkeypatch.setattr(pipeline, "unload_ollama", lambda *a, **k: calls.append(a), raising=False)
    with pytest.raises(FileNotFoundError, match="cutpoint setup --only"):
        pipeline.run_events(tmp_path, layers=("audio", "visual"))
    assert calls == []


# --------------------------------------------------------------------------- unload_ollama
class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _no_sleep(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not wait")

    monkeypatch.setattr(common.time, "sleep", boom)


def test_unload_ollama_all_unloads_fail_returns_without_waiting(monkeypatch):
    _no_sleep(monkeypatch)
    gets = []

    def get(url, **k):
        gets.append(url)
        return _Resp({"models": [{"name": "m"}]})

    def post(*a, **k):
        raise common.requests.ConnectionError("down")

    monkeypatch.setattr(common.requests, "get", get)
    monkeypatch.setattr(common.requests, "post", post)
    assert common.unload_ollama("http://x") == []
    assert len(gets) == 1


# --------------------------------------------------------------------------- 360 analyze
def _analyze_stubs(monkeypatch, tmp_path: Path) -> list[str]:

    order: list[str] = []
    clip = argparse.Namespace(lrv="a.lrv", clip_id="c1")
    monkeypatch.setattr(cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(cli, "_clips", lambda project: [clip])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: order.append("preflight"))
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: order.append("events"))
    monkeypatch.setattr(views, "run_layer", lambda clips, force=False: order.append("views") or {})
    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: order.append("analyze"))
    monkeypatch.setattr(analyze_mod, "reanalyze", lambda project: None)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "ski")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    return order


def _args(**kw) -> argparse.Namespace:
    base = {
        "project": "p",
        "only": None,
        "no_events": False,
        "views_only": False,
        "force_views": False,
        "style": None,
        "model": "m",
        "language": None,
    }
    return argparse.Namespace(**{**base, **kw})


def test_cmd_analyze_preflight_failure_stops_before_views(monkeypatch, tmp_path: Path):
    order = _analyze_stubs(monkeypatch, tmp_path)

    def fail(layers):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("cutpoint.events.pipeline.preflight", fail)
    with pytest.raises(FileNotFoundError):
        cli.cmd_analyze(_args())
    assert order == []


@pytest.mark.parametrize("flag", ["no_events", "views_only"])
def test_cmd_analyze_skips_preflight(monkeypatch, tmp_path: Path, flag: str):
    order = _analyze_stubs(monkeypatch, tmp_path)
    cli.cmd_analyze(_args(**{flag: True}))
    assert "preflight" not in order
    assert "events" not in order


# --------------------------------------------------------------------------- activity
@pytest.fixture
def ambiguous(monkeypatch):
    """Footage that matches no library profile clearly (every embedding identical)."""
    v = np.ones((1, 4), np.float32) / 2
    monkeypatch.setattr(
        activity, "_text_emb", lambda prompts: np.ones((len(prompts), 4), np.float32)
    )
    monkeypatch.setattr(activity, "keyframes", lambda project: (v, []))


@pytest.mark.parametrize("installed", [True, False])
def test_unload_only_when_clip_installed(monkeypatch, tmp_path: Path, ambiguous, installed: bool):
    unloads = []
    monkeypatch.setattr(activity, "unload_ollama", lambda *a, **k: unloads.append(1))
    monkeypatch.setattr(
        activity.importlib.util, "find_spec", lambda n: object() if installed else None
    )
    monkeypatch.setattr(activity, "default_ask", lambda: (lambda p, i, s: {}, "some-model"))
    activity.resolve(tmp_path, context="packrafting trip")
    assert len(unloads) == (1 if installed else 0)


@pytest.mark.parametrize(
    ("context", "llm", "ask", "reason", "profile"),
    [
        ("downhill mtb chest mount", True, (None, None), "unavailable", "mtb"),
        ("downhill mtb chest mount", False, None, "disabled", "mtb"),
        ("downhill mtb chest mount", True, (lambda p, i, s: {}, "m"), "no usable", "mtb"),
        # names two profiles: ambiguous, so the nearest profile stays
        ("ski touring after a hike", True, (None, None), "unavailable", None),
    ],
)
def test_context_not_applied_warns_and_uses_named_profile(
    monkeypatch, tmp_path: Path, caplog, ambiguous, context, llm, ask, reason, profile
):
    if ask is not None:
        monkeypatch.setattr(activity, "default_ask", lambda: ask)
    with caplog.at_level(logging.WARNING, logger=activity.logger.name):
        rec = activity.resolve(tmp_path, context=context, llm=llm)
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("--context not applied" in w and reason in w for w in warns)
    assert rec["source"] == "fallback"
    if profile:
        assert rec["name"] == profile
        assert rec["profile"]["name"] == profile
    else:
        assert rec["name"] not in ("ski", "hike")


def test_no_context_no_warning(tmp_path: Path, caplog, ambiguous):
    with caplog.at_level(logging.WARNING, logger=activity.logger.name):
        activity.resolve(tmp_path, llm=False)
    assert not any("--context not applied" in r.getMessage() for r in caplog.records)
