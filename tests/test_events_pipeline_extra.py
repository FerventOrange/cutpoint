"""run_events fails fast, before side effects, when the events extra is missing."""

from __future__ import annotations

import pytest

from cutpoint.events import pipeline


def _no_modules(monkeypatch):
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: None)


def test_missing_module_raises_for_model_layers(monkeypatch):
    _no_modules(monkeypatch)
    with pytest.raises(ModuleNotFoundError) as exc:
        pipeline._require_events_extra(("audio", "vlm"))
    assert exc.value.name == pipeline.EVENTS_MODULES[0]


def test_vlm_only_skips_check(monkeypatch):
    _no_modules(monkeypatch)
    pipeline._require_events_extra(("vlm",))


def test_all_modules_present_passes(monkeypatch):
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    pipeline._require_events_extra(("audio", "ocr", "vlm"))


def test_run_events_fails_before_unloading_ollama(monkeypatch, tmp_path):
    _no_modules(monkeypatch)
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: {"sources": []})
    calls = []
    monkeypatch.setattr(pipeline, "unload_ollama", lambda *a, **k: calls.append(a), raising=False)
    with pytest.raises(ModuleNotFoundError):
        pipeline.run_events(tmp_path)
    assert calls == []


def test_run_events_all_footage_offline_fails_before_side_effects(monkeypatch, tmp_path):
    hl = {"sources": [{"path": str(tmp_path / "gone.mp4"), "duration": 60.0}]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    calls = []
    monkeypatch.setattr(pipeline, "unload_ollama", lambda *a, **k: calls.append(a), raising=False)
    with pytest.raises(ValueError, match="reachable"):
        pipeline.run_events(tmp_path, layers=("audio",))
    assert calls == []
    assert not (tmp_path / "events.json").exists()
