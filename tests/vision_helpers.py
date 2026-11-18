"""Doubles shared by the vision tests: an Ollama HTTP response, a hosted provider and no Ollama."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from cutpoint import vision
from cutpoint.assistant.providers import Reply


class Resp:
    """A ``requests`` response from Ollama: ``body`` is its JSON (``bad_json``: not JSON) and a
    ``status`` of 400 or more fails ``raise_for_status``. Not a redirect unless a test says so."""

    is_redirect = False

    def __init__(self, body: Any = None, status: int = 200, *, bad_json: bool = False) -> None:
        self.body, self.status_code, self.bad_json = body, status, bad_json
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self.bad_json:
            raise ValueError("not json")
        return self.body


class FakeSubscription:
    """A hosted (subscription) provider that answers every chat with ``text``; ``sent``: the
    messages of each chat, ``kwargs``: its keyword arguments."""

    name, model, local, host = "codex", "gpt-x", False, "chatgpt.com"

    def __init__(self, text: str) -> None:
        self.text = text
        self.sent: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []

    def capabilities(self) -> None:
        pass

    def chat(self, messages, **kw) -> Reply:
        self.sent.append(messages)
        self.kwargs.append(kw)
        return Reply(self.text)


def no_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama is unreachable: any request to it fails the test."""

    def no_network(*_a, **_k):
        raise AssertionError("Ollama was asked")

    monkeypatch.setattr(vision.requests, "post", no_network)
    monkeypatch.setattr(vision.requests, "get", no_network)
