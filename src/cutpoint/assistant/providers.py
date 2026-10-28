"""Chat model providers: one interface, with what each model can do.

Messages and tool definitions use Ollama's ``/api/chat`` shapes (``role``/``content`` messages,
``{"type": "function", "function": {...}}`` tools), which are OpenAI's too. A provider needs tool
calls or JSON-schema output (:class:`Capabilities`); :mod:`cutpoint.assistant.session` uses
tool calls when the model has them and JSON mode otherwise.

:func:`make` builds the configured provider through :data:`PROVIDERS`, which tests replace.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from cutpoint.backends import ollama_url
from cutpoint.config import AssistantSettings, assistant_settings
from cutpoint.vision import default_model as vision_model

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"  # tools, vision and thinking; about 6.6 GB
AUTO_MODEL = f"{DEFAULT_OLLAMA_MODEL} (else the vision model)"  # shown until resolved
CHAT_TIMEOUT_S = 300.0  # one read of a local model's reply on a slow GPU
SHOW_TIMEOUT_S = 5.0
TEMPERATURE = 0.2


class ProviderError(RuntimeError):
    """The model could not be asked or its reply not read; the message is for the user."""


class ToolsUnsupportedError(ProviderError):
    """The model refused a request with tools: ask again in JSON mode."""


class CancelledError(ProviderError):
    """The turn was cancelled while the reply came in; the request was closed."""


@dataclass(frozen=True)
class Capabilities:
    tool_calls: bool  # native tool calls
    json_schema: bool  # replies constrained to a JSON schema


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    error: str | None = None  # the arguments were not a JSON object


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # the assistant message to add to the conversation as is (with its tool calls)
    message: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str  # "ollama"
    model: str
    local: bool  # runs on this machine: nothing leaves it
    context_tokens: int

    def capabilities(self) -> Capabilities:
        """What the model can do. Raises :class:`ProviderError` when it cannot be reached."""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply:
        """One reply to ``messages``, with ``tools`` offered or constrained to ``schema``.
        Raises :class:`CancelledError` soon after ``cancelled()`` turns true."""
        ...

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        """The message that answers ``call``."""
        ...


def describe(provider: Provider) -> dict[str, Any]:
    """What the page shows about ``provider``: ``{name, model, local, label}``."""
    where = "Local" if provider.local else provider.name.capitalize()
    return {
        "name": provider.name,
        "model": provider.model,
        "local": provider.local,
        "label": f"{where}: {provider.model}",
    }


def _arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """A tool call's arguments as an object; Ollama sends one, some models a JSON string."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return {}, "arguments are not valid JSON"
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return {}, "arguments must be a JSON object"
    return raw, None


class OllamaProvider:
    """Ollama's native ``/api/chat``, with ``tools``, ``format`` and ``options.num_ctx``."""

    name = "ollama"
    local = True

    def __init__(self, url: str, model: str, context_tokens: int) -> None:
        """``model`` "": auto, resolved by :meth:`resolve_model` on first use."""
        url = url.rstrip("/")
        self.url = url if url.startswith("http") else f"http://{url}"
        self._model = model
        self.context_tokens = context_tokens
        self._caps: Capabilities | None = None

    @property
    def model(self) -> str:
        """The model asked; :data:`AUTO_MODEL` while auto is not resolved yet."""
        return self._model or AUTO_MODEL

    def resolve_model(self) -> str:
        """The model to ask. Auto: :data:`DEFAULT_OLLAMA_MODEL` when ``/api/tags`` lists it,
        else the vision model (:func:`cutpoint.vision.default_model`); asked once."""
        if not self._model:
            try:
                resp = requests.get(f"{self.url}/api/tags", timeout=SHOW_TIMEOUT_S)
                data = resp.json() if resp.status_code < 400 else None
            except requests.RequestException as err:
                raise ProviderError(f"Ollama is not reachable at {self.url} ({err})") from err
            except ValueError:  # not JSON
                data = None
            models = data.get("models") if isinstance(data, dict) else None
            names = {m.get("name") for m in models or [] if isinstance(m, dict)}
            self._model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in names else vision_model()
        return self._model

    def _post(
        self, endpoint: str, payload: dict[str, Any], timeout: float, **kw: Any
    ) -> requests.Response:
        try:
            return requests.post(f"{self.url}/api/{endpoint}", json=payload, timeout=timeout, **kw)
        except requests.RequestException as err:
            raise ProviderError(f"Ollama is not reachable at {self.url} ({err})") from err

    @staticmethod
    def _error(resp: requests.Response) -> str:
        try:
            body = resp.json()
            text = body.get("error") if isinstance(body, dict) else None
        except ValueError:
            text = None
        return str(text or resp.text or resp.reason or "")[:300]

    def capabilities(self) -> Capabilities:
        """From ``/api/show``: tool calls when it lists ``tools``. Any Ollama model takes a
        JSON schema (``format``), so JSON mode always works."""
        if self._caps is None:
            model = self.resolve_model()
            resp = self._post("show", {"model": model}, SHOW_TIMEOUT_S)
            if resp.status_code == 404:
                raise ProviderError(
                    f"Ollama has no model {model}; pull it with `ollama pull {model}`"
                )
            if resp.status_code >= 400:
                raise ProviderError(f"Ollama answered {resp.status_code}: {self._error(resp)}")
            try:
                data = resp.json()
            except ValueError as err:
                raise ProviderError(f"Ollama sent an unreadable answer ({err})") from err
            caps = data.get("capabilities") if isinstance(data, dict) else None
            self._caps = Capabilities(isinstance(caps, list) and "tools" in caps, True)
        return self._caps

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply:
        """Streamed, so a cancelled turn closes the request at the next chunk instead of
        waiting for the whole reply."""
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": messages,
            "stream": True,
            "think": False,  # a reply in seconds, not a reasoning trace that fills num_ctx
            "options": {"num_ctx": self.context_tokens, "temperature": TEMPERATURE},
        }
        if tools:
            payload["tools"] = tools
        if schema:
            payload["format"] = schema
        resp = self._post("chat", payload, CHAT_TIMEOUT_S, stream=True)
        try:
            if resp.status_code >= 400:
                why = self._error(resp)
                if tools and "does not support tools" in why:
                    raise ToolsUnsupportedError(why)
                raise ProviderError(f"Ollama answered {resp.status_code}: {why}")
            text, raw_calls = self._read_stream(resp, cancelled)
        finally:
            resp.close()
        calls = []
        for raw in raw_calls:
            fn = raw.get("function") if isinstance(raw, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                args, error = _arguments(fn.get("arguments"))
                calls.append(ToolCall(fn["name"], args, error))
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if raw_calls:
            message["tool_calls"] = raw_calls
        return Reply(text, calls, message)

    @staticmethod
    def _read_stream(
        resp: requests.Response, cancelled: Callable[[], bool] | None
    ) -> tuple[str, list[Any]]:
        """``(text, raw tool calls)`` of a streamed reply: one JSON object per line, each with
        a piece of ``message``, the last with ``done``."""
        parts: list[str] = []
        raw_calls: list[Any] = []
        got = False
        try:
            for line in resp.iter_lines():
                if cancelled is not None and cancelled():
                    raise CancelledError("cancelled")
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError as err:
                    raise ProviderError(f"Ollama sent an unreadable answer ({err})") from err
                if not isinstance(chunk, dict):
                    raise ProviderError("Ollama sent an unreadable answer")
                if chunk.get("error"):
                    raise ProviderError(f"Ollama: {str(chunk['error'])[:300]}")
                msg = chunk.get("message")
                if isinstance(msg, dict):
                    got = True
                    if isinstance(msg.get("content"), str):
                        parts.append(msg["content"])
                    if isinstance(msg.get("tool_calls"), list):
                        raw_calls += msg["tool_calls"]
                if chunk.get("done"):
                    break
        except requests.RequestException as err:
            raise ProviderError(f"Ollama stopped answering ({err})") from err
        if not got:
            raise ProviderError("Ollama's answer has no message")
        return "".join(parts), raw_calls

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_name": call.name, "content": content}


def _ollama(settings: AssistantSettings) -> Provider:
    return OllamaProvider(
        settings.url or ollama_url(),
        settings.model,  # "": auto (OllamaProvider.resolve_model)
        settings.context_tokens,
    )


# provider name in config.json -> builder; config.ASSISTANT_PROVIDERS lists the names it accepts
PROVIDERS: dict[str, Callable[[AssistantSettings], Provider]] = {"ollama": _ollama}


def make(settings: AssistantSettings | None = None) -> Provider:
    """The configured provider (``assistant`` in ``config.json``).

    Raises:
        ValueError: If ``config.json`` is invalid, or the Ollama URL has a bad port.
    """
    settings = settings or assistant_settings()
    return PROVIDERS[settings.provider](settings)
