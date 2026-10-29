"""Chat model providers: one interface, with what each model can do.

Messages and tool definitions use Ollama's ``/api/chat`` shapes (``role``/``content`` messages,
``{"type": "function", "function": {...}}`` tools), which are OpenAI's too. A provider needs tool
calls or JSON-schema output (:class:`Capabilities`); :mod:`cutpoint.assistant.session` uses
tool calls when the model has them and JSON mode otherwise. The assistant message of a reply
and the answer to a tool call are in the provider's own shape (:attr:`Reply.message`,
:meth:`Provider.tool_result`); each provider converts the rest of the conversation itself.

Providers, all over ``requests``:

* :class:`OllamaProvider`: Ollama's ``/api/chat`` (the default).
* :class:`OpenAIProvider`: ``/v1/chat/completions`` of OpenAI or any server that speaks it
  (LM Studio, vLLM, llama-server), at the configured base URL.
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

import ipaddress
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import requests
from urllib3.exceptions import LocationParseError
from urllib3.util import parse_url

from cutpoint.assistant import keys
from cutpoint.backends import ollama_url
from cutpoint.config import AssistantSettings, assistant_settings
from cutpoint.vision import default_model as vision_model

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"  # tools, vision and thinking; about 6.6 GB
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
# Settings' OpenAI and xAI models when no list came. OpenAI's Astra first: the only one with tool
# calls over Chat Completions today; the others need the Responses API (#536)
OPENAI_MODELS = ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna")
XAI_MODELS = ("grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1")
AUTO_SERVER_MODEL = "server's default model"  # an OpenAI-compatible model not resolved yet
MAX_TOKENS = 4096  # an Anthropic reply's cap (required); a proposal is far shorter
MAX_ERROR = 300  # characters of a server's error text in a message
TITLES = {
    "ollama": "Ollama",
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic",
    "xai": "xAI-compatible",
}
OFFICIAL_TITLES = {"openai": "OpenAI", "anthropic": "Anthropic", "xai": "xAI"}  # official host
# Settings' Provider choices, in order: provider name -> title. Each uses its official base URL
# (Ollama: the configured one, cutpoint.backends.ollama_url), so no URL is typed.
PRESETS = {
    "ollama": "None (local Ollama)",
    "anthropic": "Claude (Anthropic)",
    "openai": "Codex (OpenAI)",
    "xai": "Grok (xAI)",
}
MODELS_TTL_S = 600.0  # a provider's model list is asked again after this (or on Refresh)
MAX_MODEL_PAGES = 10  # Anthropic's /v1/models pages read at most
# Ids of OpenAI-compatible models this provider cannot chat with, left out of the Model list:
# those starting with a NON_CHAT_PREFIXES entry or containing a NON_CHAT_PARTS one (speech,
# images, video, embeddings, moderation, legacy completions; *-codex needs the Responses API,
# which this provider does not use). Every other id stays.
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
    name: str  # "ollama", "openai", "anthropic", "xai"
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


def _chats(model_id: str) -> bool:
    """Whether an OpenAI-compatible model id may chat (not a :data:`NON_CHAT_PREFIXES` or
    :data:`NON_CHAT_PARTS` one)."""
    low = model_id.lower()
    return not low.startswith(NON_CHAT_PREFIXES) and not any(p in low for p in NON_CHAT_PARTS)


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
    ring = "save one in Settings (kept in the system keyring)"
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
        """The model to ask. Auto: :data:`DEFAULT_OLLAMA_MODEL` when ``/api/tags`` lists it,
        else the vision model (:func:`cutpoint.vision.default_model`); asked once."""
        if not self._model:
            names = set(self._tags())
            self._model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in names else vision_model()
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
        self, method: str, url: str, payload: dict[str, Any] | None, timeout: float
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
    """``/v1/chat/completions`` with ``tools`` or a ``json_schema`` response format, for OpenAI
    and the servers that speak it (LM Studio, vLLM, llama-server).

    A base URL with a path (``https://host/api/v1``) is used as given; a bare host gets ``/v1``.
    ``model`` "": the server's only model (``/v1/models``), else an error listing them.
    """

    name = "openai"
    default_url = OPENAI_URL

    def __init__(self, url: str, model: str, context_tokens: int, key: str | None) -> None:
        super().__init__(_openai_base(url or self.default_url), model, context_tokens, key)

    @property
    def title(self) -> str:
        return "OpenAI" if self.host == "api.openai.com" else f"OpenAI-compatible {self.host}"

    @property
    def model(self) -> str:
        return self._model or AUTO_SERVER_MODEL

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

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
        ids = [i for i in self._models() if _chats(i)]
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
        payload: dict[str, Any] = {"model": self.resolve_model(), "messages": messages}
        if tools:
            payload["tools"] = tools
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "cutpoint_reply", "schema": schema},
            }
        self._stop(cancelled)
        resp = self._send("POST", f"{self.url}/chat/completions", payload, self.timeout)
        self._stop(cancelled)
        if resp.status_code < 400:
            return self._reply(self._read(resp))
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

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
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
        if system:
            payload["system"] = system
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
        resp = self._send("POST", f"{self.url}/v1/messages", payload, self.timeout)
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
            b
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
    if (want := consent_key(provider)) is not None and want not in consent:
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
