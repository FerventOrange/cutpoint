"""An explicit --ollama-url is normalised like OLLAMA_HOST (#840)."""

from __future__ import annotations

import pytest

from cutpoint.backends import normalize_ollama_url, ollama_url
from cutpoint.vision import VisionRater

CASES = [
    ("myhost", "http://myhost:11434"),
    ("myhost:8080", "http://myhost:8080"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("::1", "http://[::1]:11434"),
    ("http://myhost", "http://myhost"),
    ("http://myhost:11434/", "http://myhost:11434"),
    ("https://myhost:443/", "https://myhost:443"),
    ("http://user:pw@myhost:1234", "http://user:pw@myhost:1234"),
    ("  myhost  ", "http://myhost:11434"),
]


@pytest.mark.parametrize(("raw", "want"), CASES)
def test_flag_equals_env(monkeypatch, raw, want):
    monkeypatch.setenv("OLLAMA_HOST", raw)
    from_env = ollama_url()
    assert from_env == want
    assert normalize_ollama_url(raw, "--ollama-url") == from_env
    monkeypatch.delenv("OLLAMA_HOST")
    assert VisionRater(raw, None).url == from_env


@pytest.mark.parametrize("raw", ["myhost:99999", "http://myhost:abc"])
def test_bad_port_raises(raw):
    with pytest.raises(ValueError, match="invalid port"):
        normalize_ollama_url(raw, "--ollama-url")
    with pytest.raises(ValueError, match="invalid port"):
        VisionRater(raw, None)
