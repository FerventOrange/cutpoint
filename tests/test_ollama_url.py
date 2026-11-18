"""Every Ollama URL is normalised like ``$OLLAMA_HOST``: ``--ollama-url`` (#840) and the Models
tab's (``vision_assistant.url``, AI Chat's Ollama provider, #857). A bare host gets port 11434,
a bad port is rejected, and the consent key is computed from the normalised URL everywhere."""

from __future__ import annotations

import pytest

from conftest import set_config
from cutpoint import vision
from cutpoint.assistant import providers
from cutpoint.backends import normalize_ollama_url, ollama_url
from cutpoint.vision import VisionRater

CASES = [
    ("myhost", "http://myhost:11434"),
    ("myhost:8080", "http://myhost:8080"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("::1", "http://[::1]:11434"),
    ("http://myhost", "http://myhost"),  # a scheme keeps the scheme's port, as in #854
    ("http://myhost:11434/", "http://myhost:11434"),
    ("https://myhost/", "https://myhost"),
    ("https://myhost:443/", "https://myhost:443"),
    ("http://user:pw@myhost:1234", "http://user:pw@myhost:1234"),
    ("  myhost  ", "http://myhost:11434"),
]


@pytest.mark.parametrize(("raw", "want"), CASES)
def test_every_ollama_url_is_normalised_like_ollama_host(home, monkeypatch, raw, want):
    monkeypatch.setenv("OLLAMA_HOST", raw)
    from_env = ollama_url()
    assert from_env == want
    assert normalize_ollama_url(raw, "--ollama-url") == from_env
    monkeypatch.delenv("OLLAMA_HOST")
    assert VisionRater(raw, None).url == from_env  # the flag
    assert providers.OllamaProvider(raw, "m", 0).url == from_env  # AI Chat's Ollama
    set_config(home, vision_assistant={"provider": "ollama", "url": raw, "consent": []})
    assert vision.VisionRater(model="m").url == from_env  # the Models tab's vision URL


def _vision_from_config(home, raw: str) -> object:
    set_config(home, vision_assistant={"provider": "ollama", "url": raw})
    return vision.VisionRater(model="m")


ENTRY_POINTS = {
    "flag": lambda home, raw: normalize_ollama_url(raw, "--ollama-url"),
    "vision-flag": lambda home, raw: VisionRater(raw, None),
    "chat": lambda home, raw: providers.OllamaProvider(raw, "m", 0),
    "vision-config": _vision_from_config,
}


@pytest.mark.parametrize(
    ("entry", "raw"),
    [
        ("flag", "myhost:99999"),
        ("flag", "http://myhost:abc"),
        ("vision-flag", "myhost:99999"),
        ("chat", "myhost:abc"),
        ("chat", "http://myhost:99999"),
        ("vision-config", "myhost:abc"),
    ],
)
def test_a_bad_port_raises(home, monkeypatch, entry, raw):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    with pytest.raises(ValueError, match="invalid port"):
        ENTRY_POINTS[entry](home, raw)


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
