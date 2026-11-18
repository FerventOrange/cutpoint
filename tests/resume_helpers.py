"""Helpers shared by test modules, moved out of test_resume.py."""

from __future__ import annotations

from typing import Any


class _Resp:
    def __init__(self, body: Any) -> None:
        self.body, self.status_code, self.is_redirect = body, 200, False

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self.body
