"""Fakes that make ``cutpoint.events.pipeline.run_events`` runnable in a test (#1085 A531)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from cutpoint.events import common, pipeline


def stub_run_events(
    monkeypatch: pytest.MonkeyPatch,
    sources: Iterable[Any],
    highlights: dict[str, Any] | None = None,
    *,
    read_highlights: bool = False,
) -> None:
    """Fake what ``run_events`` reads before and around its layers, over ``sources`` (objects
    with a ``path``): the analysis lists them (``highlights`` adds or replaces its keys),
    ``load_source`` returns each by its path, the preflight passes, no bad ranges, no
    transcript, no audio events and no GPU memory reading.

    ``read_highlights``: the project's own ``highlights.json`` is read instead (the test wrote
    it). Patch a fake again after this call to change it for one test."""
    by_path = {str(s.path): s for s in sources}
    if not read_highlights:
        hl = {"sources": [{"path": p} for p in by_path], **(highlights or {})}
        monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])
    monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
