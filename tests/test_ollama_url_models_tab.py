"""Ollama URLs from the Models tab (``vision_assistant.url``, AI Chat's Ollama provider) are
normalised like ``$OLLAMA_HOST`` and ``--ollama-url`` (#857): a bare host gets port 11434 and a
bad port is rejected. The consent key is computed from the normalised URL everywhere."""

from __future__ import annotations

import pytest

from conftest import set_config
from cutpoint import vision
from cutpoint.assistant import providers

CASES = [
    ("myhost", "http://myhost:11434"),
    ("myhost:8080", "http://myhost:8080"),
    ("http://myhost", "http://myhost"),  # a scheme keeps the scheme's port, as in #854
    ("https://myhost/", "https://myhost"),
    ("https://myhost:9", "https://myhost:9"),
]


@pytest.mark.parametrize(("raw", "want"), CASES)
def test_the_chat_provider_normalises_its_url(raw: str, want: str) -> None:
    assert providers.OllamaProvider(raw, "m", 0).url == want


@pytest.mark.parametrize("raw", ["myhost:abc", "http://myhost:99999"])
def test_the_chat_provider_rejects_a_bad_port(raw: str) -> None:
    with pytest.raises(ValueError, match="invalid port"):
        providers.OllamaProvider(raw, "m", 0)


@pytest.mark.parametrize(("raw", "want"), CASES)
def test_the_vision_url_from_config_is_normalised(home, monkeypatch, raw: str, want: str) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    set_config(home, vision_assistant={"provider": "ollama", "url": raw, "consent": []})
    assert vision.VisionRater(model="m").url == want


def test_a_bad_vision_url_port_raises(home, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    set_config(home, vision_assistant={"provider": "ollama", "url": "myhost:abc"})
    with pytest.raises(ValueError, match="invalid port"):
        vision.VisionRater(model="m")


def test_the_consent_keys_agree_for_a_bare_host(home) -> None:
    key = "vision-v2:ollama@myhost:11434"
    set_config(home, vision_assistant={"provider": "ollama", "consent": [key]})
    assert vision.ollama_allowed("myhost")
    assert vision.ollama_allowed("http://myhost:11434")
    assert not vision.ollama_allowed("myhost:80")
    assert vision._ollama_consent_key("myhost") == key
    assert providers.consent_key(providers.OllamaProvider("myhost", "m", 0)) == (
        "assistant-v3:ollama@myhost:11434"
    )
