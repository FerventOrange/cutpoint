"""Security headers and checks of ``cutpoint ui`` (#1085): the read-only media token, the page's
Content-Security-Policy, the request body cap and project-file values on a job's command line."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from cutpoint import cache, ui
from test_ui import X3_CLIPS, cli_project, fake_start, make_project

PORT = 8765
BLANK = {"X-Cutpoint-Token": ""}


def test_media_urls_carry_a_read_only_token_the_other_routes_refuse(
    ui_client: TestClient, tmp_path: Path
) -> None:
    token = ui_client.app.state.token
    pid, _ = make_project(ui_client, tmp_path, {"a.mp4": b"0123456789"})
    url = ui_client.get(f"/api/projects/{pid}").json()["sources"][0]["url"]
    media = url.split("?t=")[1]
    assert media == ui.media_token(token) and token not in url
    assert ui_client.get(url, headers=BLANK).status_code != 403  # the media route takes it
    as_header = {"X-Cutpoint-Token": media}
    assert ui_client.get(f"/api/projects/{pid}", headers=as_header).status_code == 403
    as_query = ui_client.get(f"/api/projects/{pid}", headers=BLANK, params={"t": media})
    assert as_query.status_code == 403
    assert ui_client.delete(f"/api/projects/{pid}", headers=as_header).status_code == 403
    # the session token still opens a media URL
    assert ui_client.get(url.split("?")[0], headers=BLANK, params={"t": token}).status_code != 403


def test_the_page_allows_only_its_own_scripts_and_the_inline_theme_script_by_hash(
    ui_client: TestClient,
) -> None:
    html = (ui.WEB_DIR / "index.html").read_text(encoding="utf-8")
    (inline,) = re.findall(r"<script>(.*?)</script>", html, re.S)
    digest = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
    for path in ("/", "/api/session"):
        policy = ui_client.get(path).headers["content-security-policy"]
        directives = dict(d.strip().split(" ", 1) for d in policy.split(";"))
        assert directives["script-src"] == f"'self' 'sha256-{digest}'"
        assert directives["default-src"] == "'self'"
        assert directives["frame-ancestors"] == "'none'"


def test_a_request_body_over_the_cap_is_refused_before_it_is_read(ui_client: TestClient) -> None:
    big = {"Content-Length": str(ui.MAX_BODY_BYTES + 1), "Content-Type": "application/json"}
    res = ui_client.put("/api/settings", content=b"{}", headers=big)
    assert res.status_code == 413
    assert ui_client.put("/api/settings", json={}).status_code != 413  # a usual body passes


def test_an_input_in_highlights_json_that_is_an_option_never_reaches_the_job(
    ui_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["settings"]["inputs"], hl["style"] = ["--force"], "gaming"
    cache.save_json(p / "highlights.json", hl)
    seen: list[list[str]] = []
    monkeypatch.setattr(ui.Jobs, "start", fake_start(seen))
    for body in ({"action": "analyze"}, {"action": "edit", "kind": "highlights"}):
        res = ui_client.post("/api/projects/cli/jobs", json=body)
        assert res.status_code == 400 and "'--force'" in res.json()["detail"], res.text
    assert seen == []


def test_a_360_clip_in_meta_json_that_is_not_a_number_never_reaches_the_job(
    ui_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    pid = ui_client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    meta_file = ui.project_dir(pid) / ui.META_FILE
    meta = cache.load_json(meta_file)
    cache.save_json(meta_file, {**meta, "clips": ["001", "--force"]})
    seen: list[list[str]] = []
    monkeypatch.setattr(ui.Jobs, "start", fake_start(seen))
    res = ui_client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 400 and "not clip ids" in res.json()["detail"]
    assert seen == []


def _put_chunked(app, chunks: list[bytes], token: str) -> tuple[list[int], dict, bytes]:
    """Drive the ASGI app with a body in several chunks and no Content-Length."""
    headers = [
        (b"host", f"127.0.0.1:{PORT}".encode()),
        (b"content-type", b"application/json"),
        (b"transfer-encoding", b"chunked"),
        (b"x-cutpoint-token", token.encode()),
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": "/api/settings",
        "raw_path": b"/api/settings",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", PORT),
    }
    queue = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]
    sent: list[dict] = []

    async def receive() -> dict:
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(msg: dict) -> None:
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    starts = [m for m in sent if m["type"] == "http.response.start"]
    head = {k.decode(): v.decode() for k, v in starts[0]["headers"]} if starts else {}
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return [m["status"] for m in starts], head, body


def test_a_chunked_body_over_the_cap_is_refused_as_it_is_read(ui_client: TestClient) -> None:
    chunk = b" " * (1024 * 1024)
    body = [b"{"] + [chunk] * 16 + [b"}"]  # 16 MiB + 2 bytes, no Content-Length
    starts, head, out = _put_chunked(ui_client.app, body, ui_client.app.state.token)
    assert starts == [413]
    assert b"too large" in out
    assert head["x-frame-options"] == "DENY"  # the guard's headers are on it


def test_a_chunked_body_under_the_cap_still_works(ui_client: TestClient) -> None:
    starts, _, out = _put_chunked(
        ui_client.app, [b"{", b" " * 1000, b"}"], ui_client.app.state.token
    )
    assert starts == [400]  # the route's own answer to an empty settings body
    assert b"send projects_dir" in out
