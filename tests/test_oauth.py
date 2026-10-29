"""OAuth engine: PKCE, authorize URL, callback parsing, login over a loopback redirect, refresh and
grant storage. The token endpoints are faked; nothing reaches a provider."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import threading
import time
import urllib.parse

import pytest
import requests

from cutpoint.assistant import grants, keys, oauth


class FakeRing:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, user: str) -> str | None:
        return self.items.get((service, user))

    def set_password(self, service: str, user: str, value: str) -> None:
        self.items[(service, user)] = value

    def delete_password(self, service: str, user: str) -> None:
        self.items.pop((service, user), None)


@pytest.fixture
def ring(monkeypatch: pytest.MonkeyPatch) -> FakeRing:
    fake = FakeRing()
    monkeypatch.setattr(keys, "_keyring", lambda: fake)
    return fake


class FakeResponse:
    def __init__(self, status: int, data: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._data = data
        self.text = text or (str(data) if data is not None else "")

    def json(self) -> dict:
        if self._data is None:
            raise ValueError("no json")
        return self._data


def test_pkce_challenge_is_s256_of_verifier():
    verifier, challenge = oauth.pkce()
    assert len(verifier) == 128  # 96 random bytes, base64url, no padding
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    assert challenge == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("http://localhost:54545/callback?code=abc&state=xyz", ("abc", "xyz")),
        ("abc#xyz", ("abc", "xyz")),
        ("  abc  ", ("abc", "")),
    ],
)
def test_parse_callback_input(text, expected):
    assert oauth.parse_callback_input(text) == expected


def _claude_token_response() -> dict:
    return {
        "access_token": "sk-ant-oat-ACCESS",
        "refresh_token": "REFRESH-1",
        "expires_in": 3600,
        "account": {"uuid": "acct-1", "email_address": "a@example.com"},
        "organization": {"uuid": "org-1", "name": "Org"},
    }


def _login_with_redirect(monkeypatch, provider: str, *, state_override: str | None = None):
    """Run login() with the browser replaced by a thread that hits the loopback redirect."""
    flow = dataclasses.replace(oauth.FLOWS[provider], port=0)  # ephemeral port, no clash in CI
    monkeypatch.setitem(oauth.FLOWS, provider, flow)
    posted: dict = {}

    def fake_post(url, **kwargs):
        posted.update(url=url, **kwargs)
        return FakeResponse(200, _claude_token_response())

    monkeypatch.setattr(oauth.requests, "post", fake_post)

    def fake_open(url: str) -> bool:
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        redirect = query["redirect_uri"]
        state = state_override or query["state"]

        def hit():
            requests.get(f"{redirect}?code=CODE&state={state}", timeout=10)

        threading.Thread(target=hit, daemon=True).start()
        return True

    monkeypatch.setattr(oauth.webbrowser, "open", fake_open)
    return flow, posted


def test_login_exchanges_code_and_stores_grant(monkeypatch, ring):
    _, posted = _login_with_redirect(monkeypatch, "claude-code")
    grant = oauth.login("claude-code", notify=lambda _: None, timeout=10)
    assert posted["url"] == oauth.FLOWS["claude-code"].token_url
    body = posted["json"]
    assert body["grant_type"] == "authorization_code"
    assert body["code"] == "CODE"
    assert body["client_id"] == oauth.FLOWS["claude-code"].client_id
    assert grant.access == "sk-ant-oat-ACCESS"
    assert grant.email == "a@example.com"
    assert grant.org_name == "Org"
    assert grants.load("claude-code") == grant  # round-trips through the keyring
    assert "oauth:claude-code" in {user for _, user in ring.items}


def test_login_rejects_state_mismatch_and_stores_nothing(ring):
    with pytest.raises(oauth.OAuthError, match="state did not match"):
        oauth.login(
            "claude-code", notify=lambda _: None, open_browser=False, paste=lambda: "CODE#FORGED"
        )
    assert grants.load("claude-code") is None


def test_refresh_dead_grant_when_provider_says_invalid_grant(monkeypatch, ring):
    monkeypatch.setattr(
        oauth.requests, "post", lambda *a, **k: FakeResponse(400, {"error": "invalid_grant"})
    )
    grant = grants.Grant("claude-code", "old", "REFRESH-1", expires_ms=0)
    with pytest.raises(oauth.OAuthDeadError):
        oauth.refresh(grant)


def test_refresh_transient_failure_is_not_dead(monkeypatch, ring):
    monkeypatch.setattr(oauth.requests, "post", lambda *a, **k: FakeResponse(503, text="busy"))
    grant = grants.Grant("claude-code", "old", "REFRESH-1", expires_ms=0)
    with pytest.raises(oauth.OAuthError) as err:
        oauth.refresh(grant)
    assert not isinstance(err.value, oauth.OAuthDeadError)


def test_access_token_refreshes_near_expiry_and_keeps_login_stamp(monkeypatch, ring):
    stamp = int(time.time() * 1000) - 1000
    grants.save(
        grants.Grant(
            "claude-code",
            "old",
            "REFRESH-1",
            expires_ms=int(time.time() * 1000) + 1000,
            authorized_at_ms=stamp,
        )
    )
    monkeypatch.setattr(
        oauth.requests,
        "post",
        lambda *a, **k: FakeResponse(
            200, {"access_token": "new", "refresh_token": "REFRESH-2", "expires_in": 3600}
        ),
    )
    assert grants.access_token("claude-code") == "new"
    stored = grants.load("claude-code")
    assert stored.refresh == "REFRESH-2"
    assert stored.authorized_at_ms == stamp


def test_access_token_does_not_refresh_when_fresh(monkeypatch, ring):
    grants.save(
        grants.Grant("codex", "still-good", "R", expires_ms=int(time.time() * 1000) + 3_600_000)
    )

    def boom(*a, **k):
        raise AssertionError("no refresh expected")

    monkeypatch.setattr(oauth.requests, "post", boom)
    assert grants.access_token("codex") == "still-good"


def test_anthropic_grant_past_30_days_needs_login(ring):
    old = int(time.time() * 1000) - grants.ANTHROPIC_GRANT_TTL_MS - 1000
    grants.save(grants.Grant("claude-code", "a", "r", expires_ms=0, authorized_at_ms=old))
    with pytest.raises(grants.GrantExpiredError, match="30-day"):
        grants.access_token("claude-code")


def test_status_never_contains_tokens(ring):
    grants.save(
        grants.Grant(
            "codex", "SECRET-ACCESS", "SECRET-REFRESH", expires_ms=int(time.time() * 1000) + 9999
        )
    )
    assert "SECRET" not in repr(grants.status("codex"))
    assert grants.status("antigravity") == {"provider": "antigravity", "logged_in": False}


def test_callback_server_ignores_mismatched_state():
    flow = dataclasses.replace(oauth.FLOWS["claude-code"], port=0)
    server = oauth._CallbackServer(flow, expected_state="valid-state-123")
    try:
        resp_bad = requests.get(
            f"http://127.0.0.1:{server.port}{flow.path}?code=BADCODE&state=wrong-state",
            timeout=5,
        )
        assert resp_bad.status_code == 400
        assert server.results.empty()

        resp_good = requests.get(
            f"http://127.0.0.1:{server.port}{flow.path}?code=GOODCODE&state=valid-state-123",
            timeout=5,
        )
        assert resp_good.status_code == 200
        res = server.wait(timeout=2)
        assert res["code"] == "GOODCODE"
        assert res["state"] == "valid-state-123"
    finally:
        server.stop()


def test_antigravity_project_discovery_success(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(
            200,
            {
                "cloudaicompanionProject": {"id": "discovered-proj-123"},
                "paidTier": {"id": "g1-pro-tier"},
            },
        )

    monkeypatch.setattr(oauth.requests, "post", fake_post)
    proj = oauth._antigravity_project("test-token")
    assert proj == "discovered-proj-123"
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == f"{oauth.CLOUDCODE}/v1internal:loadCodeAssist"
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["headers"]["User-Agent"] == oauth.ANTIGRAVITY_HEADERS["User-Agent"]
    assert kwargs["headers"]["Accept-Encoding"] == "gzip"
    assert kwargs["json"] == {"metadata": {"ideType": "ANTIGRAVITY"}}


def test_antigravity_project_discovery_repeats_when_paid_tier_absent(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            return FakeResponse(
                200,
                {
                    "cloudaicompanionProject": "companion-proj-456",
                },
            )
        return FakeResponse(
            200,
            {
                "cloudaicompanionProject": "companion-proj-456",
                "currentTier": {"id": "standard-tier"},
            },
        )

    monkeypatch.setattr(oauth.requests, "post", fake_post)
    proj = oauth._antigravity_project("test-token")
    assert proj == "companion-proj-456"
    assert len(calls) == 2
    assert calls[1][1]["json"] == {
        "cloudaicompanionProject": "companion-proj-456",
        "metadata": {"ideType": "ANTIGRAVITY"},
    }


def test_antigravity_project_error_raises_and_never_onboards(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return FakeResponse(403, text="PERMISSION_DENIED: account not eligible")

    monkeypatch.setattr(oauth.requests, "post", fake_post)
    with pytest.raises(oauth.OAuthError) as exc:
        oauth._antigravity_project("test-token")
    assert "Antigravity Code Assist error: 403" in str(exc.value)
    # Verifies only loadCodeAssist was called and onboardUser was never invoked
    assert all("loadCodeAssist" in url for url in calls)
    assert not any("onboardUser" in url for url in calls)


def test_antigravity_project_missing_raises_explaining_official_sign_in(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return FakeResponse(200, {"cloudaicompanionProject": ""})

    monkeypatch.setattr(oauth.requests, "post", fake_post)
    with pytest.raises(oauth.OAuthError) as exc:
        oauth._antigravity_project("test-token")
    assert "please sign into official Google Antigravity" in str(exc.value)
    assert not any("onboardUser" in url for url in calls)


def test_exchange_remembers_tokens_before_enrichment(monkeypatch):
    remembered = []
    monkeypatch.setattr(keys, "_remember", lambda val: remembered.append(val))

    monkeypatch.setattr(
        oauth,
        "_post",
        lambda flow, body: {
            "access_token": "ya29.SECRET-ACCESS-EXCHANGE",
            "refresh_token": "1//SECRET-REFRESH-EXCHANGE",
            "expires_in": 3600,
        },
    )

    def boom_enrich(flow, grant, data):
        # Verify tokens were already remembered before enrich / discovery could raise
        assert "ya29.SECRET-ACCESS-EXCHANGE" in remembered
        assert "1//SECRET-REFRESH-EXCHANGE" in remembered
        raise oauth.OAuthError("enrichment failed after tokens remembered")

    monkeypatch.setattr(oauth, "_enrich", boom_enrich)

    flow = oauth.FLOWS["antigravity"]
    with pytest.raises(oauth.OAuthError, match="enrichment failed"):
        oauth._exchange(flow, "code", "verifier", "http://127.0.0.1:51121/oauth-callback", "state")

    assert "ya29.SECRET-ACCESS-EXCHANGE" in remembered
    assert "1//SECRET-REFRESH-EXCHANGE" in remembered
