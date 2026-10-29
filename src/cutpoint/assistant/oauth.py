"""Native OAuth 2.0 authorization-code + PKCE login for the subscription providers.

One engine, three flow configs (:data:`FLOWS`): the parameters come from the vendors' public
client flows, so the same client id the vendor CLIs register is used. Nothing here calls a
provider unless :func:`login` or :func:`refresh` is invoked by the user (CLI or UI); tests run
against loopback fixtures only.

Login: build the authorize URL with a PKCE challenge and a random state, open it, catch the
redirect on a loopback server (or let the user paste the redirect URL/code), then exchange the code.
Refresh: :func:`refresh` is called by :mod:`cutpoint.assistant.grants` when the access token is
near expiry.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import logging
import queue
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import requests

from cutpoint.assistant import grants, keys

logger = logging.getLogger(__name__)
TIMEOUT_S = 300  # how long the loopback waits for the browser redirect
EXCHANGE_TIMEOUT_S = 30
# Application identifier passed in OpenAI Codex OAuth requests.
ORIGINATOR = "cutpoint"
DEAD_ERRORS = ("invalid_grant", "invalid_token", "unauthorized_client", "revoked")


# User-Agent and encoding for Antigravity / Cloud Code endpoints, matching official Antigravity Hub.
def _antigravity_user_agent() -> str:
    os_type = (
        "darwin"
        if sys.platform == "darwin"
        else ("windows" if sys.platform == "win32" else "linux")
    )
    arch = "arm64" if os_type == "darwin" else "amd64"
    return f"antigravity/hub/2.26.0 (aidev_client; os_type={os_type}; arch={arch}; cl=1005016104)"


ANTIGRAVITY_HEADERS: dict[str, str] = {
    "User-Agent": _antigravity_user_agent(),
    "Accept-Encoding": "gzip",
}


class OAuthError(RuntimeError):
    """A login or refresh failed; the message is safe to show (tokens redacted)."""


class OAuthDeadError(OAuthError):
    """The refresh token was rejected for good: the user must log in again."""


@dataclass(frozen=True)
class Flow:
    """Everything that varies per provider."""

    provider: str
    client_id: str
    authorize_url: str
    token_url: str
    scope: str
    port: int
    path: str
    callback_host: str = (
        "localhost"  # Google Antigravity uses literal 127.0.0.1; Claude/Codex use localhost
    )
    extra: dict[str, str] = field(default_factory=dict)
    json_body: bool = False  # token request as JSON (Anthropic) rather than form
    port_fallback: bool = True  # bind an ephemeral port when the fixed one is busy
    client_secret: str = ""  # Google's installed-app secret is sent on exchange and refresh
    skew_ms: int = 0  # subtract from expires_in so refresh starts early (Google: 5 min)


FLOWS: dict[str, Flow] = {
    "claude-code": Flow(
        provider="claude-code",
        client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        authorize_url="https://claude.ai/oauth/authorize",
        token_url="https://api.anthropic.com/v1/oauth/token",
        scope="org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload",
        port=54545,
        path="/callback",
        extra={"code": "true"},
        json_body=True,
    ),
    "codex": Flow(
        provider="codex",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        scope="openid profile email offline_access api.connectors.read api.connectors.invoke",
        port=1455,
        path="/auth/callback",
        extra={
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": ORIGINATOR,
        },
        port_fallback=False,  # OpenAI registers the exact port
    ),
    "antigravity": Flow(
        provider="antigravity",
        client_id="1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scope=" ".join(
            "https://www.googleapis.com/auth/" + s
            for s in (
                "cloud-platform",
                "userinfo.email",
                "userinfo.profile",
                "cclog",
                "experimentsandconfigs",
                "aicode",
            )
        ),
        port=51121,
        path="/oauth-callback",
        callback_host="127.0.0.1",
        extra={"access_type": "offline", "prompt": "consent"},
        client_secret="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",  # public installed-app secret, not a credential
        skew_ms=300_000,
    ),
}


def pkce() -> tuple[str, str]:
    """A ``(verifier, challenge)`` pair: 96 random bytes, base64url, S256 challenge."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(96)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def authorize_url(flow: Flow, redirect_uri: str, challenge: str, state: str) -> str:
    """The browser URL, parameters in the order the vendor flows use."""
    params = {
        "client_id": flow.client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": flow.scope,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        **flow.extra,
    }
    return flow.authorize_url + "?" + urllib.parse.urlencode(params)


def parse_callback_input(text: str) -> tuple[str, str]:
    """``(code, state)`` from a pasted redirect URL, or from ``code#state``. ``state`` may be ``""``."""
    text = text.strip()
    if "://" in text or "?" in text:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(text).query)
        return query.get("code", [""])[0], query.get("state", [""])[0]
    if "#" in text:
        code, _, state = text.partition("#")
        return code, state
    return text, ""


class _LoopbackHttpServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = False


class _CallbackServer:
    """Loopback listener for one redirect. Binds 127.0.0.1; the redirect URI says ``localhost``."""

    def __init__(self, flow: Flow, expected_state: str = "") -> None:
        self.results: queue.Queue[dict[str, str]] = queue.Queue()
        results, path = self.results, flow.path

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urllib.parse.urlsplit(self.path)
                if url.path != path:
                    self.send_error(404)
                    return
                query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                if expected_state and query.get("state") != expected_state:
                    self.send_error(400, "State mismatch")
                    return
                results.put(query)
                body = b"<p>Login finished. You can close this tab and return to Cutpoint.</p>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:  # keep the query string out of logs
                return

        try:
            self.server = _LoopbackHttpServer(("127.0.0.1", flow.port), Handler)
        except OSError:
            if not flow.port_fallback:
                raise OAuthError(
                    f"port {flow.port} is busy; close the other login and retry"
                ) from None
            self.server = _LoopbackHttpServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.server.shutdown()
            self.server.server_close()

    def wait(self, timeout: float) -> dict[str, str]:
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            raise OAuthError(
                "no browser redirect arrived; log in again or paste the redirect URL"
            ) from None
        finally:
            self.stop()


def login(
    provider: str,
    *,
    notify: Callable[[str], None] = print,
    open_browser: bool = True,
    paste: Callable[[], str] | None = None,
    timeout: float = TIMEOUT_S,
) -> grants.Grant:
    """Run the interactive login for ``provider`` and store the grant in the keyring.

    ``notify`` receives the authorize URL (printed for the user). ``paste``, if given, is asked for
    the redirect URL or code when no redirect arrives in time (CLI passes ``input``).

    Raises:
        OAuthError: Login was cancelled, timed out, or the exchange failed.
    """
    flow = FLOWS[provider]
    verifier, challenge = pkce()
    state = secrets.token_hex(16)
    server = _CallbackServer(flow, expected_state=state)
    redirect_uri = f"http://{flow.callback_host}:{server.port}{flow.path}"
    url = authorize_url(flow, redirect_uri, challenge, state)
    notify(url)
    if open_browser:
        webbrowser.open(url)
    query = server.wait(timeout) if paste is None else _first_of(server, timeout, paste)
    if "error" in query:
        raise OAuthError(f"the provider refused the login: {keys.redact(query['error'])}")
    code, got_state = query.get("code", ""), query.get("state", "")
    if not code:
        raise OAuthError("the redirect had no code")
    if got_state != state:
        raise OAuthError("the login state did not match; start the login again")
    grant = _exchange(flow, code, verifier, redirect_uri, state)
    grants.save(grant)
    return grant


def _first_of(server: _CallbackServer, timeout: float, paste: Callable[[], str]) -> dict[str, str]:
    """Race the loopback redirect against a pasted redirect URL or code."""
    got: queue.Queue[dict[str, str]] = queue.Queue()

    def _from_paste() -> None:
        with contextlib.suppress(Exception):
            code, state = parse_callback_input(paste())
            if code:
                got.put({"code": code, "state": state})

    def _from_server() -> None:
        with contextlib.suppress(Exception):
            res = server.results.get(timeout=timeout)
            got.put(res)

    threading.Thread(target=_from_paste, daemon=True).start()
    threading.Thread(target=_from_server, daemon=True).start()

    try:
        return got.get(timeout=timeout)
    except queue.Empty:
        raise OAuthError(
            "no browser redirect arrived; log in again or paste the redirect URL"
        ) from None
    finally:
        server.stop()


def _post(flow: Flow, body: dict[str, str]) -> dict:
    if flow.client_secret:
        body = {**body, "client_secret": flow.client_secret}
    try:
        if flow.json_body:
            resp = requests.post(
                flow.token_url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=EXCHANGE_TIMEOUT_S,
            )
        else:
            resp = requests.post(flow.token_url, data=body, timeout=EXCHANGE_TIMEOUT_S)
    except requests.RequestException as err:
        raise OAuthError(f"{flow.provider} network error: {keys.redact(str(err))}") from err
    if resp.status_code != 200:
        text = keys.redact(resp.text)[:500]
        error = ""
        with contextlib.suppress(ValueError):
            error = str(resp.json().get("error", ""))
        if resp.status_code in (400, 401) and (
            error in DEAD_ERRORS or "refresh token expired" in text.lower()
        ):
            raise OAuthDeadError(
                f"{flow.provider} refused the refresh ({resp.status_code} {error})"
            )
        raise OAuthError(f"{flow.provider} token exchange failed: {resp.status_code} {text}")
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError) as err:
        raise OAuthError(
            f"{flow.provider} returned invalid JSON: {keys.redact(resp.text)[:200]}"
        ) from err
    if not isinstance(data, dict):
        raise OAuthError(f"{flow.provider} returned an invalid response structure")
    return data


def _exchange(flow: Flow, code: str, verifier: str, redirect_uri: str, state: str) -> grants.Grant:
    body = {
        "grant_type": "authorization_code",
        "client_id": flow.client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    if flow.json_body:
        body["state"] = state  # Anthropic's exchange echoes the state
    data = _post(flow, body)
    access = data.get("access_token")
    if not access or not isinstance(access, str):
        raise OAuthError(f"{flow.provider} token exchange response missing access_token")
    refresh = data.get("refresh_token", "")
    keys._remember(access)
    if refresh:
        keys._remember(refresh)
    now_ms = int(time.time() * 1000)
    grant = grants.Grant(
        provider=flow.provider,
        access=access,
        refresh=refresh,
        expires_ms=now_ms + int(data.get("expires_in", 3600)) * 1000 - flow.skew_ms,
        authorized_at_ms=now_ms,
    )
    return _enrich(flow, grant, data)


def _enrich(flow: Flow, grant: grants.Grant, data: dict) -> grants.Grant:
    """Account metadata per provider: Claude's account block, Codex's JWT claims, Google's userinfo
    plus Antigravity's Cloud Code project."""
    if flow.provider == "claude-code":
        account = data.get("account") or {}
        org = data.get("organization") or {}
        return replace(
            grant,
            email=account.get("email_address", ""),
            account_id=account.get("uuid", ""),
            org_id=org.get("uuid", ""),
            org_name=org.get("name", ""),
        )
    if flow.provider == "codex":
        claims = _jwt_claims(grant.access)
        auth = claims.get("https://api.openai.com/auth") or {}
        profile = claims.get("https://api.openai.com/profile") or {}
        return replace(
            grant,
            account_id=auth.get("chatgpt_account_id", ""),
            email=profile.get("email", ""),
        )
    email = _google_email(grant.access)
    return replace(
        grant, email=email, project_id=_antigravity_project(grant.access), api_endpoint=CLOUDCODE
    )


CLOUDCODE = "https://daily-cloudcode-pa.googleapis.com"


def _google_email(access: str) -> str:
    try:
        resp = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
            headers={"Authorization": f"Bearer {access}"},
            timeout=EXCHANGE_TIMEOUT_S,
        )
        return resp.json().get("email", "") if resp.status_code == 200 else ""
    except (requests.RequestException, ValueError):
        return ""


def _antigravity_project(access: str) -> str:
    """Discover the existing Cloud Code project for the account via ``loadCodeAssist``.

    Uses existing entitlement only: does not auto-onboard, provision GCP projects, or accept terms.
    """
    headers = {
        **ANTIGRAVITY_HEADERS,
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json",
    }
    meta: dict[str, object] = {"metadata": {"ideType": "ANTIGRAVITY"}}
    try:
        resp = requests.post(
            f"{CLOUDCODE}/v1internal:loadCodeAssist",
            json=meta,
            headers=headers,
            timeout=EXCHANGE_TIMEOUT_S,
        )
    except requests.RequestException as err:
        raise OAuthError(f"Antigravity Code Assist network error: {keys.redact(str(err))}") from err
    if resp.status_code != 200:
        text = keys.redact(resp.text)[:500]
        raise OAuthError(f"Antigravity Code Assist error: {resp.status_code} {text}")
    try:
        data = resp.json()
    except (ValueError, json.JSONDecodeError):
        raise OAuthError("Antigravity Code Assist returned invalid JSON") from None
    if not isinstance(data, dict):
        raise OAuthError("Antigravity Code Assist returned an invalid response")
    project = _project_id(data)
    if project and not data.get("paidTier"):
        repeat_payload = {"cloudaicompanionProject": project, **meta}
        try:
            repeat_resp = requests.post(
                f"{CLOUDCODE}/v1internal:loadCodeAssist",
                json=repeat_payload,
                headers=headers,
                timeout=EXCHANGE_TIMEOUT_S,
            )
        except requests.RequestException as err:
            raise OAuthError(
                f"Antigravity Code Assist network error: {keys.redact(str(err))}"
            ) from err
        if repeat_resp.status_code != 200:
            text = keys.redact(repeat_resp.text)[:500]
            raise OAuthError(f"Antigravity Code Assist error: {repeat_resp.status_code} {text}")
        try:
            repeat_data = repeat_resp.json()
        except (ValueError, json.JSONDecodeError):
            raise OAuthError("Antigravity Code Assist returned invalid JSON") from None
        if not isinstance(repeat_data, dict):
            raise OAuthError("Antigravity Code Assist returned an invalid response")
    if not project:
        raise OAuthError(
            "no Antigravity project found; please sign into official Google Antigravity to "
            "initialize your existing entitlement (Cutpoint does not auto-provision or accept GCP terms)"
        )
    return project


def _project_id(data: dict) -> str:
    value = data.get("cloudaicompanionProject", "")
    if isinstance(value, dict):
        value = value.get("id", "")
    return value if isinstance(value, str) else ""


def _jwt_claims(token: str) -> dict:
    """The payload of a JWT, unverified (it came from the issuer over TLS a moment ago)."""
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}


def refresh(grant: grants.Grant) -> grants.Grant:
    """A refreshed grant from ``grant``'s refresh token (metadata kept, login stamp kept).

    Raises:
        OAuthDeadError: The provider rejected the refresh token; log in again.
        OAuthError: Any other failure (transient; the stored grant is left alone).
    """
    flow = FLOWS[grant.provider]
    body = {
        "grant_type": "refresh_token",
        "client_id": flow.client_id,
        "refresh_token": grant.refresh,
    }
    data = _post(flow, body)
    access = data.get("access_token")
    if not access or not isinstance(access, str):
        raise OAuthError(f"{flow.provider} refresh response missing access_token")
    now_ms = int(time.time() * 1000)
    expires_ms = now_ms + int(data.get("expires_in", 3600)) * 1000 - flow.skew_ms
    fresh = grants.merge_refresh(grant, access, data.get("refresh_token", ""), expires_ms)
    if flow.provider == "codex":
        claims = _jwt_claims(fresh.access)
        auth = claims.get("https://api.openai.com/auth") or {}
        fresh = replace(fresh, account_id=auth.get("chatgpt_account_id", fresh.account_id))
    if flow.provider == "antigravity" and not fresh.project_id:
        raise OAuthError("the Antigravity grant has no project id; log in again")
    return fresh
