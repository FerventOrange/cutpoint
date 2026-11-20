"""Helpers shared by test modules, moved out of test_resume.py and test_runstate.py."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cutpoint import cache, cli, progress, runstate


class _Resp:
    def __init__(self, body: Any) -> None:
        self.body, self.status_code, self.is_redirect = body, 200, False

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self.body


def make_project(home: Path, source: str) -> Path:
    """The project ``p`` under ``home`` with one source, ``source``, in its highlights.json."""
    p = home / "projects" / "p"
    p.mkdir(parents=True)
    cache.save_json(p / "highlights.json", {"sources": [{"path": source}]})
    return p


def read_progress(project: Path) -> dict:
    """The project's progress.json as it is on disk now."""
    return json.loads((project / runstate.FILE).read_text(encoding="utf-8"))


def fake_events(
    monkeypatch: pytest.MonkeyPatch,
    body: Callable[[Path], Any] | None = None,
    steps: tuple[str, ...] = ("vlm",),
) -> list[dict]:
    """``cutpoint events`` as a fake that runs the progress ``steps`` in order, then ``body``
    (given the project) inside the last one. Returns one dict per call: ``cwd``, ``args`` (the
    parsed namespace), ``record`` (progress.json at the start of the command) and ``last_step``
    (progress.json inside the last step, before ``body``)."""
    calls: list[dict] = []

    def cmd(a):
        project = runstate._ACTIVE.project
        call = {"cwd": Path.cwd(), "record": read_progress(project), "args": a}
        calls.append(call)
        with progress.run(steps) as run:
            for name in steps:
                with run.step(name):
                    if name == steps[-1]:
                        call["last_step"] = read_progress(project)
                        if body is not None:
                            body(project)

    monkeypatch.setattr(cli, "_cmd_events", cmd)
    return calls
