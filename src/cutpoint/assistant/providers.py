"""Chat model providers: one interface, with what each model can do.

Messages and tool definitions use Ollama's ``/api/chat`` shapes (``role``/``content`` messages,
``{"type": "function", "function": {...}}`` tools), which are OpenAI's too. A provider needs tool
calls or JSON-schema output (:class:`Capabilities`); :mod:`cutpoint.assistant.session` uses
tool calls when the model has them and JSON mode otherwise. The assistant message of a reply
and the answer to a tool call are in the provider's own shape (:attr:`Reply.message`,
:meth:`Provider.tool_result`); each provider converts the rest of the conversation itself.

Providers, all over ``requests``:

* :class:`OllamaProvider`: Ollama's ``/api/chat`` (the default).
* :class:`OpenAIProvider`: OpenAI's ``/v1/responses`` at ``api.openai.com``, else
  ``/v1/chat/completions`` of any server that speaks it (LM Studio, vLLM, llama-server,
  OpenRouter), at the configured base URL.
* :class:`XAIProvider`: the same API at xAI (Grok), whose own key variable is ``XAI_API_KEY``.
* :class:`AnthropicProvider`: Anthropic's Messages API.

Settings offers each as a preset (:data:`PRESETS`): the provider with its official base URL, so
users choose "Claude" or "Grok" instead of typing a URL. A configured ``url`` that is not the
preset's (:func:`is_preset`) still works and is shown as custom. :func:`model_list` lists the
models a provider offers for the Model field, cached for :data:`MODELS_TTL_S`.

A provider is ``local`` when its URL's host is this machine (``localhost``, a loopback address);
any other host is hosted, and the page asks the user once per :func:`consent_key` before project
text is sent there. Keys come from :mod:`cutpoint.assistant.keys` and never leave the provider
object except in its request headers.

:func:`make` builds the configured provider through :data:`PROVIDERS`, which tests replace.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import requests
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

from cutpoint.assistant import grants, keys, oauth
from cutpoint.backends import ollama_url
from cutpoint.config import AssistantSettings, assistant_settings
from cutpoint.vision import default_model as vision_model

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"  # tools, vision and thinking; about 6.6 GB
PREVIOUS_VISION_MODEL = "qwen3-vl:8b"  # the default vision model before qwen3.5:9b
AUTO_MODEL = f"{DEFAULT_OLLAMA_MODEL} or the vision model"  # shown until resolved
CHAT_TIMEOUT_S = 300.0  # one read of a local model's reply on a slow GPU
HOSTED_TIMEOUT_S = 180.0  # a hosted model's whole reply (not streamed)
CONNECT_TIMEOUT_S = 10.0
SHOW_TIMEOUT_S = 5.0
CHECK_TIMEOUT_S = 15.0  # a hosted provider's model list or model
LOCAL_CONTEXT_TOKENS = 8192  # assistant.context_tokens unset: a model on this machine
HOSTED_CONTEXT_TOKENS = 32768  # assistant.context_tokens unset: a hosted model
OPENAI_URL = "https://api.openai.com/v1"
XAI_URL = "https://api.x.ai/v1"
ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
ANTHROPIC_MODELS = (DEFAULT_ANTHROPIC_MODEL, "claude-opus-5-5")
# Settings' OpenAI and xAI models when no list came, OpenAI's Astra first. Sol, Terra and Luna make
# tool calls only over the Responses API, which api.openai.com is asked through (#536)
OPENAI_MODELS = ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna")
XAI_MODELS = ("grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1")
AUTO_SERVER_MODEL = "server's default model"  # an OpenAI-compatible model not resolved yet
MAX_TOKENS = 4096  # an Anthropic reply's cap (required); a proposal is far shorter
MAX_ERROR = 300  # characters of a server's error text in a message
TITLES = {
    "ollama": "Ollama",
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic API",
    "xai": "xAI-compatible",
    "claude-code": "Claude Code subscription",
    "codex": "Codex subscription",
    "antigravity": "Antigravity (Gemini)",
}
OFFICIAL_TITLES = {
    "openai": "OpenAI API",
    "anthropic": "Anthropic API",
    "xai": "xAI",
}  # official host
# Settings' Provider choices, in order: provider name -> title. Each uses its official base URL
# (Ollama: the configured one, cutpoint.backends.ollama_url), so no URL is typed.
PRESETS = {
    "ollama": "None (local Ollama)",
    "anthropic": "Anthropic API",
    "openai": "OpenAI API",
    "xai": "Grok (xAI)",
}
MODELS_TTL_S = 600.0  # a provider's model list is asked again after this (or on Refresh)
MAX_MODEL_PAGES = 10  # Anthropic's /v1/models pages read at most
# Ids of OpenAI-compatible models this provider cannot chat with, left out of the Model list:
# those starting with a NON_CHAT_PREFIXES entry or containing a NON_CHAT_PARTS one (speech,
# images, video, embeddings, moderation, legacy completions; *-codex needs the Responses API,
# so it stays only at api.openai.com, which is asked through it). Every other id stays.
NON_CHAT_PREFIXES = (
    "text-embedding-",
    "tts-",
    "whisper-",
    "dall-e-",
    "omni-moderation-",
    "text-moderation-",
    "gpt-image-",
    "sora-",
    "davinci-",
    "babbage-",
)
NON_CHAT_PARTS = ("-tts", "-transcribe", "-realtime", "-audio", "-image", "imagine", "codex")
# Words of a server's refusal of tools (Ollama, vLLM, llama-server without --jinja).
_NO_TOOLS = ("support", "requires", "jinja", "enable")


class ProviderError(RuntimeError):
    """The model could not be asked or its reply not read; the message is for the user."""


class ToolsUnsupportedError(ProviderError):
    """The model refused a request with tools: ask again in JSON mode."""


class CancelledError(ProviderError):
    """The turn was cancelled while the reply came in; the request was closed."""


class AmbiguousURLError(ProviderError, ValueError):
    """A base URL whose host depends on the parser (:func:`unambiguous`): nothing is sent.
    A ValueError too, so it is reported as an invalid setting."""


@dataclass(frozen=True)
class Capabilities:
    tool_calls: bool  # native tool calls
    json_schema: bool  # replies constrained to a JSON schema


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    error: str | None = None  # the arguments were not a JSON object
    id: str = ""  # the provider's id of the call (OpenAI, Anthropic), which its result names


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # the assistant message to add to the conversation as is (with its tool calls)
    message: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str  # "ollama", "openai", "anthropic", "xai", "claude-code", "codex"
    model: str
    local: bool  # runs on this machine: nothing leaves it
    host: str  # the host of its URL ("" when it has none)
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
    """What the page shows about ``provider``: ``{name, model, local, host, label}``."""
    title = TITLES.get(provider.name, provider.name.capitalize())
    if keys.official(provider.name, provider.host):
        title = OFFICIAL_TITLES[provider.name]
    where = "this machine" if provider.local else provider.host
    return {
        "name": provider.name,
        "model": provider.model,
        "local": provider.local,
        "host": provider.host,
        # "OpenAI-compatible · server's default model · openrouter.ai"
        "label": " · ".join(part for part in (title, provider.model, where) if part),
    }


def _with_scheme(url: str) -> str:
    return url if "://" in url else f"http://{url}"


def host_of(url: str) -> str:
    """The lower-case host of ``url`` (``http://`` assumed without a scheme); "" for none.

    Parsed as ``requests`` (urllib3) parses it, so the host is the one a request goes to."""
    try:
        host = parse_url(_with_scheme(url)).host or ""
    except (LocationParseError, ValueError):
        return ""
    return host.strip("[]").lower()  # an IPv6 address without its brackets


def has_userinfo(url: str) -> bool:
    """Whether ``url`` has a user name or password (``https://name@host``), by either parser."""
    try:
        if parse_url(_with_scheme(url)).auth is not None:
            return True
    except (LocationParseError, ValueError):
        pass
    try:
        return "@" in urlsplit(_with_scheme(url)).netloc
    except ValueError:
        return False


def unambiguous(url: str) -> bool:
    """Whether ``url`` has no backslash, no user name or password, and the standard library and
    urllib3 (what requests sends with) read the same host from it:
    ``https://evil.example\\@api.openai.com`` is read as ``api.openai.com`` by one and goes to
    ``evil.example`` by the other, and ``https://api.openai.com@evil.example`` reads as one
    host to a person and goes to another."""
    if "\\" in url or has_userinfo(url):
        return False
    try:
        split = (urlsplit(_with_scheme(url)).hostname or "").lower()
    except ValueError:
        return False
    return split == host_of(url)


def is_local(url: str) -> bool:
    """Whether ``url`` is on this machine: ``localhost`` (or ``*.localhost``) or a loopback
    address (``127.0.0.0/8``, ``::1``). Anything else, a LAN address too, counts as hosted."""
    host = host_of(url)
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def consent_key(provider: Provider) -> str | None:
    """What the user agrees to once before project text goes to ``provider``:
    ``<name>@<host>``; None for a local one (nothing leaves the machine)."""
    return None if provider.local else f"{provider.name}@{provider.host}"


def check(provider: Provider) -> str | None:
    """Why ``provider`` cannot run the assistant (unreachable, no key, no model, or neither tool
    calls nor JSON-schema output), or None when it can."""
    try:
        caps = provider.capabilities()
    except ProviderError as err:
        return str(err)
    if not (caps.tool_calls or caps.json_schema):
        return f"{provider.model} supports neither tool calls nor JSON output; choose another model"
    return None


def missing_key(provider: Provider) -> str | None:
    """Why ``provider`` cannot be asked for want of an API key (how to add one), or None."""
    need = getattr(provider, "_need_key", None)
    if need is None:
        return None
    try:
        need()
    except ProviderError as err:
        return str(err)
    return None


KEY_CHECKED_LATER = "The key is checked when you send the first message."


def consent_note(provider: Provider) -> str:
    """Why Settings' Save did not check ``provider``: it would contact (with its key, if any) a
    host the user has not agreed to yet (#504)."""
    what = (
        "the connection and the key are"
        if getattr(provider, "has_key", False)
        else ("the connection is")
    )
    return (
        f"Nothing was sent to {provider.host} yet: agree in the AI Chat tab to use it; {what} "
        "checked with your first message there."
    )


def check_note(provider: Provider) -> str | None:
    """What a passed :func:`check` did not prove: a key for a host other than the provider's
    own API host (OpenRouter, a LAN server) may have been unused, as such a host can list its
    models without one. None when there is nothing to add."""
    if getattr(provider, "has_key", False) and not keys.official(provider.name, provider.host):
        return KEY_CHECKED_LATER
    return None


def _redirect(title: str, resp: requests.Response, setting: str) -> ProviderError | None:
    """The error for a redirect answer (never followed: requests would resend the key, and on
    307/308 the body, to the new address), or None for any other answer."""
    if not 300 <= resp.status_code < 400:
        return None
    to = host_of((getattr(resp, "headers", None) or {}).get("Location") or "")
    return ProviderError(
        f"{title} answered {resp.status_code}, redirecting to {to or 'another address'}; "
        f"set that as the {setting} in Settings"
    )


def _http_url(url: str) -> str:
    url = url.strip().rstrip("/")
    return url if "://" in url else f"http://{url}"


def _chats(model_id: str, responses: bool = False) -> bool:
    """Whether an OpenAI-compatible model id may chat (not a :data:`NON_CHAT_PREFIXES` or
    :data:`NON_CHAT_PARTS` one; ``codex`` is one only without the Responses API)."""
    low = model_id.lower()
    parts = [p for p in NON_CHAT_PARTS if not (responses and p == "codex")]
    return not low.startswith(NON_CHAT_PREFIXES) and not any(p in low for p in parts)


def _openai_base(url: str) -> str:
    """The base URL an OpenAI-compatible provider uses: as given with a path, else ``/v1``."""
    base = _http_url(url)
    return base + "/v1" if urlsplit(base).path in ("", "/") else base


_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


def public(text: str) -> str:
    """``text`` with the ``user:password@`` part of every URL in it left out: what a message
    (a turn's error, which ``chat.json`` keeps) may show."""
    return _USERINFO.sub("", text)


def _key_hint(name: str, host: str) -> str:
    """How to add ``name``'s key for ``host``: its own variable only for its own host,
    :data:`keys.ENV_KEY` only for a host that is not official (#504), or the keyring."""
    ring = "save one in the Models tab (kept in the system keyring)"
    names = keys.env_names(name, host)
    if not names:
        return ring
    note = ""
    if names[0] != keys.ENV_KEY and keys.has_shared():  # set, but never sent to this host
        note = f" ({keys.ENV_KEY} is only for custom base URLs)"
    return f"set {names[0]} before starting cutpoint ui{note}, or {ring}"


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

    def __init__(self, url: str, model: str, context_tokens: int) -> None:
        """``model`` "": auto, resolved by :meth:`resolve_model` on first use."""
        url = url.rstrip("/")
        self.url = url if url.startswith("http") else f"http://{url}"
        self.host = host_of(self.url)
        self.local = is_local(self.url)  # an Ollama on another machine is hosted (consent)
        self._model = model
        self.context_tokens = context_tokens
        self._caps: Capabilities | None = None

    @property
    def model(self) -> str:
        """The model asked; :data:`AUTO_MODEL` while auto is not resolved yet."""
        return self._model or AUTO_MODEL

    def resolve_model(self) -> str:
        """The model to ask. Auto: the first of :data:`DEFAULT_OLLAMA_MODEL`, the vision model
        (:func:`cutpoint.vision.default_model`) and :data:`PREVIOUS_VISION_MODEL` that
        ``/api/tags`` lists, else :data:`DEFAULT_OLLAMA_MODEL`; asked once."""
        if not self._model:
            names = set(self._tags())
            wanted = (DEFAULT_OLLAMA_MODEL, vision_model(), PREVIOUS_VISION_MODEL)
            self._model = next((m for m in wanted if m in names), DEFAULT_OLLAMA_MODEL)
        return self._model

    def _tags(self) -> list[str]:
        """The names of the pulled models (``/api/tags``); [] for an error answer."""
        try:
            resp = requests.get(
                f"{self.url}/api/tags", timeout=SHOW_TIMEOUT_S, allow_redirects=False
            )
            if moved := _redirect("Ollama", resp, "URL"):
                raise moved
            data = resp.json() if resp.status_code < 400 else None
        except requests.RequestException as err:
            why = public(f"Ollama is not reachable at {self.url} ({err})")
            raise ProviderError(why) from err
        except ValueError:  # not JSON
            data = None
        models = data.get("models") if isinstance(data, dict) else None
        return [
            m["name"]
            for m in models or []
            if isinstance(m, dict) and isinstance(m.get("name"), str)
        ]

    def list_models(self) -> list[dict[str, str]]:
        """The pulled models, ``{id, name}``, for the Model field."""
        return [{"id": n, "name": n} for n in self._tags()]

    def _post(
        self, endpoint: str, payload: dict[str, Any], timeout: float, **kw: Any
    ) -> requests.Response:
        try:
            resp = requests.post(
                f"{self.url}/api/{endpoint}",
                json=payload,
                timeout=timeout,
                allow_redirects=False,
                **kw,
            )
        except requests.RequestException as err:
            why = public(f"Ollama is not reachable at {self.url} ({err})")
            raise ProviderError(why) from err
        if moved := _redirect("Ollama", resp, "URL"):
            resp.close()
            raise moved
        return resp

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
            "options": {"num_ctx": self.context_tokens},
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


class _HttpProvider:
    """What the OpenAI-compatible and Anthropic adapters share: URL, key, timeouts, errors."""

    name = ""

    def __init__(self, url: str, model: str, context_tokens: int, key: str | None) -> None:
        if not unambiguous(url):  # a hand-edited config.json: Settings refuses these
            raise AmbiguousURLError(
                f"{TITLES[self.name]}: the base URL {public(url)!r} has a backslash, a user "
                "name or password, or an unclear host; fix assistant.url in Settings"
            )
        self.url = url
        self.host = host_of(url)
        self.local = is_local(url)
        self._model = model
        self.context_tokens = context_tokens or (
            LOCAL_CONTEXT_TOKENS if self.local else HOSTED_CONTEXT_TOKENS
        )
        self._key = key or None  # only ever sent in the request headers
        self.timeout = CHAT_TIMEOUT_S if self.local else HOSTED_TIMEOUT_S
        self._caps: Capabilities | None = None

    def __repr__(self) -> str:  # never the key
        return f"{type(self).__name__}({public(self.url)!r}, {self._model!r})"

    @property
    def title(self) -> str:
        return TITLES[self.name]

    @property
    def model(self) -> str:
        return self._model

    @property
    def has_key(self) -> bool:
        """Whether a key is sent (never the key)."""
        return self._key is not None

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _need_key(self) -> None:
        """A hosted server needs a key; a local one (LM Studio, llama-server) may not."""
        if self._key is None and not self.local:
            raise ProviderError(f"No API key for {self.title}: {_key_hint(self.name, self.host)}")

    def _send(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        timeout: float,
        stream: bool = False,
    ) -> requests.Response:
        """One request. A key is never sent in clear text to another machine."""
        if self._key and not self.local and urlsplit(url).scheme.lower() != "https":
            raise ProviderError(
                f"{self.title}: use an https:// base URL to send an API key "
                f"(it is not sent over http:// to {self.host})"
            )
        send = requests.post if method == "POST" else requests.get
        try:
            # never followed: requests would resend the key (and, on 307/308, the body) there
            resp = send(
                url,
                json=payload,
                headers=self._headers(),
                timeout=(CONNECT_TIMEOUT_S, timeout),
                allow_redirects=False,
                stream=stream,
            )
        except requests.Timeout as err:
            raise ProviderError(f"{self.title} did not answer within {timeout:g} s") from err
        except requests.RequestException as err:
            why = keys.redact(public(f"{self.title} is not reachable at {self.url} ({err})"))
            raise ProviderError(why) from err
        if moved := _redirect(self.title, resp, "base URL"):
            raise moved
        return resp

    @staticmethod
    def _why(resp: requests.Response) -> str:
        """The error text of an answer: ``{"error": {"message"}}`` (OpenAI, Anthropic),
        ``{"error": "..."}``, else the body; keys redacted."""
        try:
            body = resp.json()
        except ValueError:
            body = None
        text: Any = None
        if isinstance(body, dict):
            err = body.get("error")
            text = err.get("message") if isinstance(err, dict) else err
            text = text or body.get("message") or body.get("detail")
        return keys.redact(str(text or resp.text or resp.reason or ""))[:MAX_ERROR]

    def _fail(self, resp: requests.Response, why: str | None = None) -> ProviderError:
        code = resp.status_code
        why = self._why(resp) if why is None else why
        if code in (401, 403):
            return ProviderError(
                f"{self.title} rejected the API key ({code}: {why}); check "
                + "".join(f"{n} or " for n in keys.env_names(self.name, self.host))
                + "the key saved in Settings"
            )
        if code == 404:
            return ProviderError(
                f"{self.title} has no model {self.model} at {public(self.url)} (404: {why}); "
                "check the model and base URL in Settings"
            )
        if code == 429:
            return ProviderError(
                f"{self.title} is rate limiting or out of credit (429: {why}); try again later"
            )
        if code >= 500:
            return ProviderError(f"{self.title} answered {code}: {why}; try again later")
        return ProviderError(f"{self.title} answered {code}: {why}")

    def _read(self, resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as err:
            raise ProviderError(f"{self.title} sent an unreadable answer ({err})") from err
        if not isinstance(data, dict):
            raise ProviderError(f"{self.title} sent an unreadable answer")
        return data

    @staticmethod
    def _stop(cancelled: Callable[[], bool] | None) -> None:
        """A hosted reply is not streamed: a cancelled turn stops before and after it."""
        if cancelled is not None and cancelled():
            raise CancelledError("cancelled")


class OpenAIProvider(_HttpProvider):
    """OpenAI's ``/v1/responses`` at ``api.openai.com`` (its current models make tool calls only
    there, #536), else ``/v1/chat/completions`` for the servers that speak it (LM Studio, vLLM,
    llama-server, OpenRouter), with ``tools`` or a ``json_schema`` reply format.

    A base URL with a path (``https://host/api/v1``) is used as given; a bare host gets ``/v1``.
    ``model`` "": the server's only model (``/v1/models``), else an error listing them.
    """

    name = "openai"
    default_url = OPENAI_URL

    def __init__(self, url: str, model: str, context_tokens: int, key: str | None) -> None:
        super().__init__(_openai_base(url or self.default_url), model, context_tokens, key)

    @property
    def title(self) -> str:
        return "OpenAI API" if self.host == "api.openai.com" else f"OpenAI-compatible {self.host}"

    @property
    def model(self) -> str:
        return self._model or AUTO_SERVER_MODEL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

    @property
    def responses(self) -> bool:
        """Whether the Responses API is used: OpenAI's own host only (xAI and compatible
        servers speak Chat Completions)."""
        return self.name == "openai" and keys.official("openai", self.host)

    def _models(self) -> list[str]:
        resp = self._send("GET", f"{self.url}/models", None, CHECK_TIMEOUT_S)
        if resp.status_code >= 400:
            raise self._fail(resp)
        rows = self._read(resp).get("data")
        return [r["id"] for r in rows or [] if isinstance(r, dict) and isinstance(r.get("id"), str)]

    def list_models(self) -> list[dict[str, str]]:
        """The chat models, ``{id, name}`` (the API gives no display name): every id but those
        starting with one of :data:`NON_CHAT_PREFIXES` or containing one of
        :data:`NON_CHAT_PARTS`."""
        self._need_key()
        ids = [i for i in self._models() if _chats(i, self.responses)]
        return [{"id": i, "name": i} for i in sorted(ids)]

    def resolve_model(self) -> str:
        if not self._model:
            self._need_key()
            ids = self._models()
            if len(ids) != 1:
                listed = f"; it lists {', '.join(ids[:8])}" if ids else ""
                raise ProviderError(f"Choose a model for {self.title} in Settings{listed}")
            self._model = ids[0]
        return self._model

    def capabilities(self) -> Capabilities:
        """Reached with the key (``/v1/models``) and a model chosen. Tool calls are assumed: a
        server that refuses them is asked again in JSON mode (:class:`ToolsUnsupportedError`)."""
        if self._caps is None:
            self._need_key()
            if self._model:
                self._models()  # reachable, and the key accepted
            else:
                self.resolve_model()
            self._caps = Capabilities(True, True)
        return self._caps

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply:
        self._need_key()
        if self.responses:
            endpoint, payload = "responses", self._responses_payload(messages, tools, schema)
        else:
            endpoint = "chat/completions"
            payload = {"model": self.resolve_model(), "messages": messages}
            if tools:
                payload["tools"] = tools
            if schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "cutpoint_reply", "schema": schema},
                }
        self._stop(cancelled)
        resp = self._send("POST", f"{self.url}/{endpoint}", payload, self.timeout)
        self._stop(cancelled)
        if resp.status_code < 400:
            data = self._read(resp)
            return self._responses_reply(data) if self.responses else self._reply(data)
        why = self._why(resp)
        low = why.lower()
        if tools and "tool" in low and any(w in low for w in _NO_TOOLS):
            raise ToolsUnsupportedError(why)
        raise self._fail(resp, why)

    def _reply(self, data: dict[str, Any]) -> Reply:
        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        msg = first.get("message") if isinstance(first, dict) else None
        if not isinstance(msg, dict):
            raise ProviderError(f"{self.title}'s answer has no message")
        content = msg.get("content")
        if isinstance(content, list):  # content parts
            content = "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
        text = content if isinstance(content, str) else ""
        calls: list[ToolCall] = []
        raw_calls: list[dict[str, Any]] = []
        raw_list = msg.get("tool_calls")
        for n, raw in enumerate(raw_list if isinstance(raw_list, list) else [], 1):
            fn = raw.get("function") if isinstance(raw, dict) else None
            if not (isinstance(fn, dict) and isinstance(fn.get("name"), str)):
                continue
            cid = raw.get("id") if isinstance(raw.get("id"), str) and raw["id"] else f"call_{n}"
            args, error = _arguments(fn.get("arguments"))
            calls.append(ToolCall(fn["name"], args, error, cid))
            arguments = fn.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {})
            raw_calls.append(
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": arguments},
                }
            )
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if raw_calls:
            message["tool_calls"] = raw_calls
        return Reply(text, calls, message)

    def _responses_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """A ``/v1/responses`` request, not stored: system messages as ``instructions``, the rest
        as input items; function tools flattened and non-strict, as Chat Completions takes them."""
        instructions: list[str] = []
        items: list[dict[str, Any]] = []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if "type" in m:  # a function_call_output (tool_result)
                items.append(m)
            elif role == "system":
                if isinstance(content, str) and content.strip():
                    instructions.append(content)
            elif role == "assistant" and isinstance(m.get("output"), list):
                items += m["output"]  # a reply's items as sent, reasoning and calls included
            elif role == "user" and isinstance(content, list):  # text and image parts
                parts = [_responses_part(p) for p in content if isinstance(p, dict)]
                items.append({"role": "user", "content": parts})
            elif role in ("user", "assistant") and isinstance(content, str) and content.strip():
                item = {"role": role, "content": content}
                if role == "assistant":  # an earlier reply: OpenAI asks for its phase
                    item["phase"] = "final_answer"
                items.append(item)
        # store false: OpenAI keeps no copy of the chat; its reasoning items then carry
        # encrypted_content by default, so they replay without include
        payload: dict[str, Any] = {"model": self.resolve_model(), "input": items, "store": False}
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object"},
                    "strict": False,
                }
                for t in tools
                if isinstance(fn := t.get("function"), dict)
            ]
        if schema:
            payload["text"] = {
                "format": {"type": "json_schema", "name": "cutpoint_reply", "schema": schema}
            }
        return payload

    def _responses_reply(self, data: dict[str, Any]) -> Reply:
        """A ``/v1/responses`` answer: the text of its ``message`` items and its
        ``function_call`` items. :attr:`Reply.message` keeps every output item for the next
        request, which sends them back as they came (reasoning items go back with the calls)."""
        output = data.get("output")
        if not isinstance(output, list):
            raise ProviderError(f"{self.title}'s answer has no output")
        if data.get("status") == "incomplete":  # a tool call cut off midway is not one to run
            details = data.get("incomplete_details")
            why = details.get("reason") if isinstance(details, dict) else None
            raise ProviderError(
                f"{self.title}'s reply was cut off ({why or 'incomplete'}); try a simpler request"
            )
        texts: list[tuple[Any, str]] = []  # (phase, text) of each message item
        refused = False
        calls: list[ToolCall] = []
        items = [i for i in output if isinstance(i, dict) and isinstance(i.get("type"), str)]
        for item in items:
            if item["type"] == "message" and isinstance(item.get("content"), list):
                parts: list[str] = []
                for c in item["content"]:
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        parts.append(c["text"])
                    refused = refused or c.get("type") == "refusal"
                if parts:
                    texts.append((item.get("phase"), "".join(parts)))
            elif item["type"] == "function_call" and isinstance(item.get("name"), str):
                args, error = _arguments(item.get("arguments"))
                calls.append(ToolCall(item["name"], args, error, str(item.get("call_id") or "")))
        final = [t for phase, t in texts if phase == "final_answer"]  # not the commentary
        text = "\n\n".join(final or [t for _, t in texts])
        if refused and not text and not calls:
            raise ProviderError(f"{self.title} declined this request")
        return Reply(text, calls, {"role": "assistant", "content": text, "output": items})

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        if self.responses:
            return {"type": "function_call_output", "call_id": call.id, "output": content}
        return {"role": "tool", "tool_call_id": call.id, "content": content}


class XAIProvider(OpenAIProvider):
    """xAI's (Grok's) OpenAI-compatible API at :data:`XAI_URL`, with its own key
    (``XAI_API_KEY`` or the keyring's ``xai`` entry, only for ``api.x.ai``)."""

    name = "xai"
    default_url = XAI_URL

    @property
    def title(self) -> str:
        return "xAI" if keys.official(self.name, self.host) else f"xAI-compatible {self.host}"


class AnthropicProvider(_HttpProvider):
    """Anthropic's Messages API (``/v1/messages``) with tool use.

    System messages become ``system``; consecutive messages of one role are merged, so the
    results of one reply's tool calls go back in one user message.
    """

    name = "anthropic"

    def __init__(self, url: str, model: str, context_tokens: int, key: str | None) -> None:
        super().__init__(_http_url(url or ANTHROPIC_URL), model, context_tokens, key)

    @property
    def model(self) -> str:
        return self._model or DEFAULT_ANTHROPIC_MODEL

    def _headers(self) -> dict[str, str]:
        out = {"anthropic-version": ANTHROPIC_VERSION}
        if self._key:
            out["x-api-key"] = self._key
        return out

    def capabilities(self) -> Capabilities:
        """The model exists and the key is accepted (``/v1/models/<model>``); every Claude
        model has tool use."""
        if self._caps is None:
            self._need_key()
            url = f"{self.url}/v1/models/{quote(self.model, safe='')}"
            resp = self._send("GET", url, None, CHECK_TIMEOUT_S)
            if resp.status_code >= 400:
                raise self._fail(resp)
            self._caps = Capabilities(True, False)
        return self._caps

    def list_models(self) -> list[dict[str, str]]:
        """The models, ``{id, name}`` (``display_name``), from ``/v1/models``: every page
        (``has_more``, ``after_id``), at most :data:`MAX_MODEL_PAGES`."""
        self._need_key()
        out: list[dict[str, str]] = []
        after = ""
        for _ in range(MAX_MODEL_PAGES):
            url = f"{self.url}/v1/models?limit=1000"
            if after:
                url += f"&after_id={quote(after, safe='')}"
            resp = self._send("GET", url, None, CHECK_TIMEOUT_S)
            if resp.status_code >= 400:
                raise self._fail(resp)
            data = self._read(resp)
            rows = [
                r
                for r in data.get("data") or []
                if isinstance(r, dict) and isinstance(r.get("id"), str)
            ]
            for r in rows:
                name = r.get("display_name")
                out.append({"id": r["id"], "name": name if isinstance(name, str) else r["id"]})
            last = data.get("last_id")
            if data.get("has_more") is not True or not rows:
                break
            after = last if isinstance(last, str) and last else rows[-1]["id"]
        return out

    def _messages_url(self) -> str:
        return f"{self.url}/v1/messages"

    def _system(self, text: str) -> Any:
        return text

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,  # unused: in JSON mode the rules ask for JSON
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply:
        self._need_key()
        system, wire = _anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": wire,
        }
        system_payload = self._system(system)
        if system_payload:
            payload["system"] = system_payload
        if tools:
            payload["tools"] = [
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object"},
                }
                for t in tools
                if isinstance(fn := t.get("function"), dict)
            ]
        self._stop(cancelled)
        resp = self._send("POST", self._messages_url(), payload, self.timeout)
        self._stop(cancelled)
        if resp.status_code >= 400:
            raise self._fail(resp)
        data = self._read(resp)
        stop = data.get("stop_reason")
        if stop == "max_tokens":  # a tool call cut off midway is not one to run
            raise ProviderError(
                f"{self.title}'s reply was cut off at {MAX_TOKENS} tokens; try a simpler request"
            )
        if stop == "refusal":
            raise ProviderError(f"{self.title} declined this request")
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError(f"{self.title}'s answer has no content")
        parts: list[str] = []
        calls: list[ToolCall] = []
        blocks: list[dict[str, Any]] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
                blocks.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "tool_use" and isinstance(b.get("name"), str):
                cid = str(b.get("id") or "")
                args, error = _arguments(b.get("input"))
                calls.append(ToolCall(b["name"], args, error, cid))
                blocks.append({"type": "tool_use", "id": cid, "name": b["name"], "input": args})
        return Reply("".join(parts), calls, {"role": "assistant", "content": blocks})

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        block = {"type": "tool_result", "tool_use_id": call.id, "content": content}
        return {"role": "user", "content": [block]}


def _anthropic_block(b: dict[str, Any]) -> dict[str, Any]:
    """A content block as the Messages API takes it: the neutral image part
    ``{"type": "image", "media_type", "data" (base64)}`` becomes a base64 ``source``."""
    if b.get("type") != "image":
        return b
    source = {"type": "base64", "media_type": b["media_type"], "data": b["data"]}
    return {"type": "image", "source": source}


def _responses_part(p: dict[str, Any]) -> dict[str, Any]:
    """A neutral text or image part as a Responses input part (an image as a data URL)."""
    if p.get("type") == "image":
        return {"type": "input_image", "image_url": f"data:{p['media_type']};base64,{p['data']}"}
    return {"type": "input_text", "text": str(p.get("text") or "")}


def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """``(system, messages)`` for the Messages API: system text joined, each message as content
    blocks without empty text, consecutive messages of one role merged."""
    system: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            if isinstance(content, str) and content.strip():
                system.append(content)
            continue
        if role not in ("user", "assistant"):
            continue
        raw = [{"type": "text", "text": content}] if isinstance(content, str) else content
        blocks = [
            _anthropic_block(b)
            for b in (raw if isinstance(raw, list) else [])
            if isinstance(b, dict)
            and not (b.get("type") == "text" and not str(b.get("text") or "").strip())
        ]
        if not blocks:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] = [*out[-1]["content"], *blocks]
        else:
            out.append({"role": role, "content": blocks})
    return "\n\n".join(system), out


def _ollama(settings: AssistantSettings) -> Provider:
    return OllamaProvider(
        settings.url or ollama_url(),
        settings.model,  # "": auto (OllamaProvider.resolve_model)
        settings.context_tokens or LOCAL_CONTEXT_TOKENS,
    )


CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_CODE_BETA = "oauth-2025-04-20"
CODEX_URL = "https://chatgpt.com/backend-api/codex"
CODEX_CLIENT_VERSION = "0.162.1"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_ANTIGRAVITY_MODEL = "gemini-3.1-pro"


SUBSCRIPTION_HOSTS = {
    "claude-code": "api.anthropic.com",
    "codex": "chatgpt.com",
}


def is_subscription_official(provider: str, host: str) -> bool:
    h = (host or "").lower()
    if provider == "antigravity":
        return h == "googleapis.com" or h.endswith(".googleapis.com")
    return SUBSCRIPTION_HOSTS.get(provider) == h


class _Subscription:
    """What the subscription providers share: the login grant (``cutpoint login NAME``) is the
    bearer token, not an API key."""

    name = ""

    def _token(self) -> str:
        try:
            return grants.access_token(self.name)
        except (grants.GrantExpiredError, oauth.OAuthError, keys.KeyringError) as err:
            raise ProviderError(str(err)) from err

    def _need_key(self) -> None:
        grant = grants.load(self.name)
        if grant is None:
            with contextlib.suppress(Exception):
                if grants.access_token(self.name):
                    return
            raise ProviderError(f"{self.title}: run cutpoint login {self.name} first")
        end = grants.deadline_ms(grant)
        now_ms = time.time() * 1000
        if end is not None and now_ms >= end:
            raise ProviderError(
                f"{self.title} login expired (30-day limit); run cutpoint login {self.name} again"
            )
        expires_ms = getattr(grant, "expires_ms", None)
        if expires_ms is not None and now_ms >= expires_ms and not getattr(grant, "refresh", ""):
            raise ProviderError(
                f"{self.title} login expired and has no refresh token; run cutpoint login {self.name} again"
            )

    def _send(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        timeout: float,
        stream: bool = False,
    ) -> requests.Response:
        dest_host = host_of(url)
        if not is_subscription_official(self.name, dest_host):
            raise ProviderError(
                f"{self.title}: host {dest_host!r} is not allowed for {self.name} subscription"
            )
        if urlsplit(url).scheme.lower() != "https":
            raise ProviderError(f"{self.title}: use an https:// base URL to send the login token")
        return super()._send(method, url, payload, timeout, stream)

    def _fail(self, resp: requests.Response, why: str | None = None) -> ProviderError:
        if resp.status_code in (401, 403):
            return ProviderError(
                f"{self.title} refused the login ({resp.status_code}); "
                f"run cutpoint login {self.name} again"
            )
        return super()._fail(resp, why)


class ClaudeCodeProvider(_Subscription, AnthropicProvider):
    """Claude on a Claude subscription: Messages at ``api.anthropic.com`` with the login's bearer
    token and the Claude Code identity as the first system block."""

    name = "claude-code"

    def __init__(self, url: str, model: str, context_tokens: int) -> None:
        super().__init__(url, model, context_tokens, None)

    @property
    def title(self) -> str:
        return "Claude Code subscription"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token()}",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": CLAUDE_CODE_BETA,
        }

    def _messages_url(self) -> str:
        return f"{self.url}/v1/messages?beta=true"

    def _system(self, text: str) -> list[dict[str, str]]:
        blocks = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]
        if text:
            blocks.append({"type": "text", "text": text})
        return blocks

    def capabilities(self) -> Capabilities:
        self._token()
        return Capabilities(tool_calls=True, json_schema=False)


class CodexProvider(_Subscription, OpenAIProvider):
    """ChatGPT's Codex backend on a ChatGPT subscription: Responses, streamed (the backend takes
    no other form), with the login's bearer token and account id."""

    name = "codex"
    default_url = CODEX_URL

    def __init__(self, url: str, model: str, context_tokens: int) -> None:
        super().__init__(url, model, context_tokens, None)
        self._session = str(uuid.uuid4())

    @property
    def title(self) -> str:
        return "Codex subscription"

    @property
    def model(self) -> str:
        return self._model or DEFAULT_CODEX_MODEL

    @property
    def responses(self) -> bool:
        return True

    def resolve_model(self) -> str:
        return self.model

    def list_models(self) -> list[dict[str, str]]:
        self._need_key()
        grant = grants.load(self.name)
        if grant is None or not grant.account_id:
            raise ProviderError(f"{self.title}: run cutpoint login {self.name} first")
        url = f"{self.url.rstrip('/')}/models?client_version={CODEX_CLIENT_VERSION}"
        resp = self._send("GET", url, None, CHECK_TIMEOUT_S)
        if resp.status_code >= 400:
            raise self._fail(resp)
        data = self._read(resp)
        rows = data.get("models")
        if not isinstance(rows, list):
            rows = data.get("data")
        if not isinstance(rows, list):
            raise ProviderError(f"{self.title}'s model list has no models")
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            slug = r.get("slug") or r.get("id")
            if not isinstance(slug, str) or not slug.strip():
                continue
            slug = slug.strip()
            if r.get("supported_in_api") is False:
                continue
            visibility = str(r.get("visibility") or "").strip().lower()
            if visibility in ("hide", "hidden"):
                continue
            if slug in seen:
                continue
            seen.add(slug)
            display = r.get("display_name")
            name = display.strip() if isinstance(display, str) and display.strip() else slug
            out.append({"id": slug, "name": name})
        return out

    def capabilities(self) -> Capabilities:
        self._token()
        return Capabilities(tool_calls=True, json_schema=True)

    def _headers(self) -> dict[str, str]:
        grant = grants.load(self.name)
        if grant is None or not grant.account_id:
            raise ProviderError(f"{self.title}: run cutpoint login {self.name} first")
        return {
            "Authorization": f"Bearer {self._token()}",
            "chatgpt-account-id": grant.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": oauth.ORIGINATOR,
            "session-id": self._session,
        }

    def _completed(self, resp: requests.Response) -> dict[str, Any]:
        """The final response of a streamed answer: its ``response.completed`` event."""
        resp.encoding = "utf-8"  # SSE is UTF-8 even when the backend omits Content-Type.
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not (isinstance(line, str) and line.startswith("data:")):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind == "response.completed" and isinstance(event.get("response"), dict):
                    return event["response"]
                if kind in ("response.failed", "error"):
                    why = event.get("error") or event.get("message") or kind
                    raise ProviderError(f"{self.title} failed: {keys.redact(str(why))[:MAX_ERROR]}")
            raise ProviderError(f"{self.title}'s answer ended without a completed response")
        except requests.RequestException as err:
            raise ProviderError(f"{self.title} stream error: {keys.redact(str(err))}") from err
        finally:
            resp.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply:
        self._need_key()
        payload = self._responses_payload(messages, tools, schema)
        payload.update(stream=True, include=["reasoning.encrypted_content"])
        self._stop(cancelled)
        resp = self._send("POST", f"{self.url}/responses", payload, self.timeout, stream=True)
        self._stop(cancelled)
        if resp.status_code >= 400:
            why = self._why(resp)
            low = why.lower()
            if tools and "tool" in low and any(w in low for w in _NO_TOOLS):
                raise ToolsUnsupportedError(why)
            raise self._fail(resp, why)
        return self._responses_reply(self._completed(resp))


def _gemini_parts(content: Any) -> list[dict[str, Any]]:
    """Gemini parts of a neutral message: a string, or a list of ``text`` and ``image`` parts."""
    raw = [{"type": "text", "text": content}] if isinstance(content, str) else content
    parts: list[dict[str, Any]] = []
    for p in raw if isinstance(raw, list) else []:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text" and str(p.get("text") or "").strip():
            parts.append({"text": p["text"]})
        elif p.get("type") == "image":
            parts.append({"inlineData": {"mimeType": p["media_type"], "data": p["data"]}})
    return parts


def _gemini_contents(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """``(system, contents)`` for Cloud Code: system text joined, roles ``user``/``model``,
    consecutive messages of one role merged (the results of one reply's calls go in one turn).
    A message with ``parts`` is already Gemini's (a :attr:`Reply.message` or a tool result)."""
    system: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            if isinstance(content, str) and content.strip():
                system.append(content)
            continue
        if role not in ("user", "assistant", "model"):
            continue
        parts = m["parts"] if isinstance(m.get("parts"), list) else _gemini_parts(content)
        if not parts:
            continue
        role = "user" if role == "user" else "model"
        if out and out[-1]["role"] == role:
            out[-1]["parts"] = [*out[-1]["parts"], *parts]
        else:
            out.append({"role": role, "parts": parts})
    return "\n\n".join(system), out


class AntigravityProvider(_Subscription, _HttpProvider):
    """Antigravity account quota through Cloud Code Assist's streamed Gemini wire protocol."""

    name = "antigravity"

    def __init__(self, url: str, model: str, context_tokens: int) -> None:
        grant = grants.load(self.name)
        base = url or (grant.api_endpoint if grant else "") or oauth.CLOUDCODE
        super().__init__(_http_url(base), model, context_tokens, None)
        self._agent_id = str(uuid.uuid4())
        self._trajectory_id = str(uuid.uuid4())
        self._session_id = f"-{uuid.uuid4().int % (9 * 10**18) + 1}"
        self._step = 0
        self._request_count = 0

    @property
    def title(self) -> str:
        return "Antigravity (Gemini)"

    @property
    def model(self) -> str:
        return self._model or DEFAULT_ANTIGRAVITY_MODEL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token()}", **oauth.ANTIGRAVITY_HEADERS}

    def list_models(self) -> list[dict[str, str]]:
        grant = grants.load(self.name)
        if grant is None or not grant.project_id:
            raise ProviderError(f"{self.title}: run cutpoint login {self.name} first")
        resp = self._send(
            "POST",
            f"{self.url}/v1internal:fetchAvailableModels",
            {"project": grant.project_id},
            CHECK_TIMEOUT_S,
        )
        if resp.status_code >= 400:
            raise self._fail(resp)
        data = self._read(resp)
        models = data.get("models")
        if not isinstance(models, dict):
            raise ProviderError(f"{self.title}'s model list has no models")
        image_models = data.get("imageGenerationModelIds") or []
        return [
            {"id": name, "name": row.get("displayName") or name}
            for name, row in models.items()
            if isinstance(row, dict)
            and not row.get("disabled")
            and not row.get("isInternal")
            and name not in image_models
            and name not in ("chat_20706", "chat_23310", "gemini-2.5-pro")
        ]

    def resolve_model(self) -> str:
        if not self._model:
            names = [row["id"] for row in self.list_models()]
            self._model = (
                DEFAULT_ANTIGRAVITY_MODEL
                if DEFAULT_ANTIGRAVITY_MODEL in names
                else next((name for name in names if name.startswith("gemini-")), "")
            )
            if not self._model:
                raise ProviderError(f"{self.title}: this account offers no Gemini chat model")
        return self.model

    def _stream_response(
        self, resp: requests.Response, cancelled: Callable[[], bool] | None
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        finish = ""
        try:
            for line in resp.iter_lines():
                self._stop(cancelled)
                if not line.startswith(b"data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError as err:
                    raise ProviderError(f"{self.title} sent an unreadable stream event") from err
                if not isinstance(event, dict):
                    raise ProviderError(f"{self.title} sent an invalid stream event")
                if event.get("error"):
                    why = keys.redact(str(event["error"]))[:MAX_ERROR]
                    raise ProviderError(f"{self.title} failed: {why}")
                body = event.get("response")
                candidates = body.get("candidates") if isinstance(body, dict) else None
                if not isinstance(candidates, list) or not candidates:
                    continue
                first = candidates[0]
                if not isinstance(first, dict):
                    raise ProviderError(f"{self.title} sent an invalid candidate")
                content = first.get("content")
                if isinstance(content, dict) and isinstance(content.get("parts"), list):
                    parts.extend(content["parts"])
                finish = first.get("finishReason") or finish
        except requests.RequestException as err:
            raise ProviderError(f"{self.title}'s response stream was interrupted") from err
        finally:
            resp.close()
        if not finish:
            raise ProviderError(f"{self.title}'s answer ended without a completed response")
        return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}]}

    def capabilities(self) -> Capabilities:
        self._token()
        return Capabilities(tool_calls=True, json_schema=False)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,  # unused: in JSON mode the rules ask for JSON
        cancelled: Callable[[], bool] | None = None,
    ) -> Reply:
        self._need_key()
        grant = grants.load(self.name)
        if grant is None or not grant.project_id:
            raise ProviderError(f"{self.title}: run cutpoint login {self.name} first")
        model = self.resolve_model()
        system, contents = _gemini_contents(messages)
        request: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": MAX_TOKENS},
        }
        if system:
            request["systemInstruction"] = {"role": "user", "parts": [{"text": system}]}
        if tools:
            request["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters") or {"type": "object"},
                        }
                        for t in tools
                        if isinstance(fn := t.get("function"), dict)
                    ]
                }
            ]
        self._step = max(self._step + 1, len(contents))
        used_claude = str(model.startswith("claude-")).lower()
        request["sessionId"] = self._session_id
        request["labels"] = {
            "cascade_id": self._agent_id,
            "last_step_index": str(self._step - 1),
            "request_id": f"{self._trajectory_id}-{self._request_count}",
            "root_cascade_id": self._agent_id,
            "trajectory_id": self._trajectory_id,
            "used_claude": used_claude,
            "used_claude_conservative": used_claude,
            "used_non_gemini_model": str(not model.startswith("gemini-")).lower(),
        }
        self._request_count += 1
        payload = {
            "model": model,
            "project": grant.project_id,
            "request": request,
            "userAgent": "antigravity",
            "requestId": (
                f"agent/{self._agent_id}/{int(time.time() * 1000)}/"
                f"{self._trajectory_id}/{self._step}"
            ),
        }
        self._stop(cancelled)
        resp = self._send(
            "POST",
            f"{self.url}/v1internal:streamGenerateContent?alt=sse",
            payload,
            self.timeout,
            stream=True,
        )
        self._stop(cancelled)
        if resp.status_code >= 400:
            raise self._fail(resp)
        body = self._stream_response(resp, cancelled)
        candidates = body.get("candidates") if isinstance(body, dict) else None
        first = candidates[0] if isinstance(candidates, list) and candidates else None
        if not isinstance(first, dict):
            raise ProviderError(f"{self.title} declined this request")
        if first.get("finishReason") == "MAX_TOKENS":  # a call cut off midway is not one to run
            raise ProviderError(
                f"{self.title}'s reply was cut off at {MAX_TOKENS} tokens; try a simpler request"
            )
        content = first.get("content")
        raw_parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(raw_parts, list):
            raise ProviderError(f"{self.title}'s answer has no content")
        texts: list[str] = []
        calls: list[ToolCall] = []
        kept: list[dict[str, Any]] = []  # sent back as they came (a call's signature included)
        for part in raw_parts:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            call = part.get("functionCall")
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
                kept.append(part)
            elif isinstance(call, dict) and isinstance(call.get("name"), str):
                cid = str(call.get("id") or f"call_{len(calls) + 1}")
                args, error = _arguments(call.get("args"))
                calls.append(ToolCall(call["name"], args, error, cid))
                kept.append({**part, "functionCall": {**call, "id": cid}})
        return Reply("".join(texts), calls, {"role": "model", "parts": kept})

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        reply = {"name": call.name, "id": call.id, "response": {"result": content}}
        return {"role": "user", "parts": [{"functionResponse": reply}]}


def _claude_code(settings: AssistantSettings) -> Provider:
    return ClaudeCodeProvider(settings.url, settings.model, settings.context_tokens)


def _antigravity(settings: AssistantSettings) -> Provider:
    return AntigravityProvider(settings.url, settings.model, settings.context_tokens)


def _codex(settings: AssistantSettings) -> Provider:
    return CodexProvider(settings.url, settings.model, settings.context_tokens)


DEFAULT_URLS = {"openai": OPENAI_URL, "anthropic": ANTHROPIC_URL, "xai": XAI_URL}


def key_host(name: str, url: str) -> str:
    """The host a ``name`` provider with base URL ``url`` ("": its default) sends its key to:
    what its key is looked up and saved for (:func:`cutpoint.assistant.keys.lookup`)."""
    return host_of(_http_url(url or DEFAULT_URLS.get(name, "")))


def is_preset(name: str, url: str) -> bool:
    """Whether base URL ``url`` of provider ``name`` is its preset's (:data:`PRESETS`): "" or
    the official base URL itself (Ollama: the configured Ollama URL). Any other is custom."""
    if not url:
        return True
    if name == "ollama":
        try:
            default = ollama_url()
        except (ValueError, TypeError):  # a bad OLLAMA_HOST or config.json
            return False
        return _http_url(url).lower() == _http_url(default).lower()
    default = DEFAULT_URLS.get(name)
    if not default:
        return False
    norm = _http_url if name == "anthropic" else _openai_base
    return norm(url).lower() == default.lower()


def _openai(settings: AssistantSettings) -> Provider:
    key = keys.lookup("openai", key_host("openai", settings.url))[0]
    return OpenAIProvider(settings.url, settings.model, settings.context_tokens, key)


def _xai(settings: AssistantSettings) -> Provider:
    key = keys.lookup("xai", key_host("xai", settings.url))[0]
    return XAIProvider(settings.url, settings.model, settings.context_tokens, key)


def _anthropic(settings: AssistantSettings) -> Provider:
    key = keys.lookup("anthropic", key_host("anthropic", settings.url))[0]
    return AnthropicProvider(settings.url, settings.model, settings.context_tokens, key)


# provider name in config.json -> builder; config.ASSISTANT_PROVIDERS lists the names it accepts
PROVIDERS: dict[str, Callable[[AssistantSettings], Provider]] = {
    "ollama": _ollama,
    "openai": _openai,
    "anthropic": _anthropic,
    "claude-code": _claude_code,
    "codex": _codex,
    "antigravity": _antigravity,
    "xai": _xai,
}


_PRESET_CLASSES: dict[str, type[_HttpProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "xai": XAIProvider,
}


def preset(name: str, context_tokens: int = 0) -> Provider:
    """Preset ``name`` (:data:`PRESETS`) at its official base URL, chosen in Settings but not
    saved: its key is only its own (its variable or keyring entry for the official host), never
    :data:`keys.ENV_KEY`, which may be meant for the saved provider's host."""
    if name == "ollama":
        return _ollama(AssistantSettings("ollama", context_tokens=context_tokens))
    key = keys.lookup(name, key_host(name, ""), shared=False)[0]
    return _PRESET_CLASSES[name]("", "", context_tokens, key)  # type: ignore[return-value]


def make(settings: AssistantSettings | None = None) -> Provider:
    """The configured provider (``assistant`` in ``config.json``), with its key.

    Raises:
        ValueError: If ``config.json`` is invalid, or the Ollama URL has a bad port.
    """
    settings = settings or assistant_settings()
    return PROVIDERS[settings.provider](settings)


# (provider name, base URL) -> (when it was read, its models); successes only
_MODELS: dict[tuple[str, str], tuple[float, list[dict[str, str]]]] = {}
_MODELS_LOCK = threading.Lock()
_clock = time.monotonic  # replaced by tests


def clear_model_cache(provider: str | None = None) -> None:
    """Evict cached model lists for ``provider`` (or all providers if None)."""
    with _MODELS_LOCK:
        if provider is None:
            _MODELS.clear()
        else:
            keys_to_del = [k for k in _MODELS if k[0] == provider]
            for k in keys_to_del:
                del _MODELS[k]


def model_list(
    provider: Provider, consent: tuple[str, ...], *, refresh: bool = False
) -> dict[str, Any]:
    """The models ``provider`` offers, for Settings' Model field: ``{state, models, message}``.

    ``state`` is ``"ok"`` (``models``: ``{id, name}``), ``"key_needed"`` or ``"consent_needed"``
    (a hosted provider is asked only with its key and once the user agreed to it, ``consent``
    of ``config.json``; nothing is sent otherwise), ``"unsupported"`` or ``"error"``
    (``message`` says why, keys redacted): the field then takes any typed name. A list is kept
    for :data:`MODELS_TTL_S` per provider and base URL; ``refresh`` asks again.
    """
    lister = getattr(provider, "list_models", None)
    if lister is None:
        return {"state": "unsupported", "models": [], "message": None}
    if missing := missing_key(provider):
        return {"state": "key_needed", "models": [], "message": missing}
    is_sub_official = isinstance(provider, _Subscription) and is_subscription_official(
        provider.name, provider.host
    )
    if not is_sub_official and (want := consent_key(provider)) is not None and want not in consent:
        return {
            "state": "consent_needed",
            "models": [],
            "message": (
                f"The model list loads once this provider is saved and you agree in AI Chat to "
                f"use {provider.host}."
            ),
        }
    cache_key = (provider.name, str(getattr(provider, "url", "")))
    now = _clock()
    with _MODELS_LOCK:
        hit = _MODELS.get(cache_key)
    if hit is not None and not refresh and now - hit[0] < MODELS_TTL_S:
        return {"state": "ok", "models": list(hit[1]), "message": None}
    try:
        models = lister()
    except ProviderError as err:
        return {"state": "error", "models": [], "message": keys.redact(public(str(err)))}
    with _MODELS_LOCK:
        _MODELS[cache_key] = (now, models)
    return {"state": "ok", "models": list(models), "message": None}
