"""Job doubles shared by the UI tests: a ``Jobs.start`` spy that records the jobs it is given."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from cutpoint import ui


def job_args(args: list[str]) -> list[str]:
    """The CLI args of a job as ``start_job`` built them, before the ``--json`` it always adds
    (so the envelope's error code and outputs reach the job)."""
    assert args[-1] == "--json", args
    return args[:-1]


class JobSpy:
    """Stands in for ``Jobs.start``: records each job and reports it "running", starting nothing.

    ``args``: the CLI args of each job (without ``--json``); ``on_success``: the hook each job
    registered; ``calls``: everything each was given, as a dict (``action``, ``raw`` the args with
    ``--json``, ``args``, ``on_success``, ``check``, ``plan``, ``before`` and the keywords such as
    ``cwd`` and ``on_result``). All three are filled as the jobs start.
    ``check``: run the job's check after recording it, as the jobs lock does before a job starts
    (False: a start that skips it). ``refuse``: raised instead, before anything is recorded (as a
    second job is refused while one runs).
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        check: bool = True,
        refuse: HTTPException | None = None,
    ) -> None:
        self.args: list[list[str]] = []
        self.on_success: list[Callable[[], None] | None] = []
        self.calls: list[dict[str, Any]] = []
        self.check = check
        self.refuse = refuse

        def start(
            _jobs: object,
            project_id: str,
            action: str,
            args: list[str],
            on_success: Callable[[], None] | None = None,
            check: Callable[[], None] | None = None,
            plan: object = None,
            before: object = None,
            **kw: Any,
        ) -> SimpleNamespace:
            if self.refuse is not None:
                raise self.refuse
            self.args.append(job_args(args))
            self.on_success.append(on_success)
            self.calls.append(
                {
                    "action": action,
                    "raw": args,
                    "args": self.args[-1],
                    "on_success": on_success,
                    "check": check,
                    "plan": plan,
                    "before": before,
                    **kw,
                }
            )
            if self.check and check is not None:
                check()
            return SimpleNamespace(view=lambda: {"status": "running"})

        monkeypatch.setattr(ui.Jobs, "start", start)
