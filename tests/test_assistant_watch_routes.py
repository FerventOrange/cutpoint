"""The frame consent of ``view_moment`` in ``cutpoint ui`` (#640): per project and per host,
after the assistant's own consent; a provider on this machine needs none.

The model is the scripted ``FakeProvider``; frames are synthetic (no ffmpeg).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("httpx")
pytest.importorskip("cv2")
from fastapi.testclient import TestClient

from conftest import set_config
from cutpoint import ui
from cutpoint.assistant import watch
from test_assistant import FakeProvider, Reply, call
from test_assistant_routes import HostedFake, ask, use
from test_assistant_routes import client as client  # the fixtures
from test_assistant_routes import no_keys as no_keys
from test_assistant_routes import proj as proj
from test_assistant_watch import jpeg

ASSISTANT = "assistant-v2:anthropic@api.anthropic.com"
FRAMES = "frames-v1:anthropic@api.anthropic.com"
URL = "/api/projects/cli/assistant/frames-consent"


@pytest.fixture(autouse=True)
def frames(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    grabbed: list[float] = []

    def grab(_src, t):
        grabbed.append(t)
        return jpeg()

    monkeypatch.setattr(watch, "grab_frame", grab)
    monkeypatch.setattr(watch, "_TEXT_ONLY", set())
    return grabbed


def images(fake: FakeProvider) -> int:
    return sum(
        1
        for r in fake.requests
        for m in r["messages"]
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if part.get("type") == "image"
    )


def test_a_local_provider_needs_no_frame_consent(client: TestClient, monkeypatch, proj) -> None:
    p, seq, _ids = proj
    fake = use(monkeypatch, FakeProvider([call("view_moment", id="m1"), Reply("A fish.")]))
    got = client.get("/api/projects/cli/assistant").json()
    assert got["frames"] == {"needed": False, "key": None, "host": "localhost"}
    assert client.post(URL, json={"key": None}).status_code == 409
    view = ask(client, "what is in m1?", seq)
    assert view["status"] == "done" and images(fake) == 1
    assert any(e.get("name") == "view_moment" for e in view["events"])
    assert len(list((p / "ui" / "assistant" / "sheets").iterdir())) == 1


def test_hosted_frames_need_their_own_consent_per_project(
    client: TestClient, monkeypatch, proj, home: Path, frames: list[float]
) -> None:
    p, seq, _ids = proj
    fake = use(
        monkeypatch,
        HostedFake([call("view_moment", id="m1"), Reply("Not allowed yet.")]),
    )
    # the frame consent comes after the assistant's own
    assert client.post(URL, json={"key": FRAMES}).status_code == 409
    set_config(home, assistant={"provider": "ollama", "consent": [ASSISTANT]})
    got = client.get("/api/projects/cli/assistant").json()
    assert got["consent"]["needed"] is False
    assert got["frames"] == {"needed": True, "key": FRAMES, "host": "api.anthropic.com"}

    view = ask(client, "what is in m1?", seq)  # assistant consent only: no frame leaves
    assert view["status"] == "done" and images(fake) == 0 and frames == []
    ev = next(e for e in view["events"] if e["kind"] == "frames_consent")
    assert ev["key"] == FRAMES and ev["host"] == "api.anthropic.com"
    assert "not allowed video frames" in fake.requests[1]["messages"][-1]["content"]

    assert client.post(URL, json={"key": "frames-v1:anthropic@elsewhere"}).status_code == 409
    for _ in range(2):  # allowing again keeps one entry
        res = client.post(URL, json={"key": FRAMES})
        assert res.status_code == 200, res.text
        assert res.json()["frames"]["needed"] is False
    saved = json.loads((home / "config.json").read_text("utf-8"))
    assert saved["assistant"]["consent"] == [ASSISTANT]  # untouched
    assert saved["assistant"]["frames_consent"] == {watch.project_key(p): [FRAMES]}

    fake.script = [call("view_moment", id="m1"), Reply("A fish.")]
    fake.requests.clear()
    view = ask(client, "and now?", seq)
    assert view["status"] == "done" and images(fake) == 1 and len(frames) == 6

    # another project of the same host is not allowed by it
    other = ui.projects_dir() / "other"
    other.mkdir()
    (other / "highlights.json").write_bytes((p / "highlights.json").read_bytes())
    got = client.get("/api/projects/other/assistant").json()
    assert got["frames"]["needed"] is True


def test_a_broken_frames_consent_section_is_never_overwritten(
    client: TestClient, monkeypatch, proj, home: Path
) -> None:
    use(monkeypatch, HostedFake([]))
    set_config(home, assistant={"consent": [ASSISTANT], "frames_consent": ["x"]})
    assert client.get("/api/projects/cli/assistant").json()["frames"]["needed"] is True
    res = client.post(URL, json={"key": FRAMES})
    assert res.status_code == 409 and "frames_consent" in res.json()["detail"]
    saved = json.loads((home / "config.json").read_text("utf-8"))
    assert saved["assistant"]["frames_consent"] == ["x"]


def test_the_frame_consent_route_takes_the_token_header_only(client: TestClient, proj) -> None:
    token = client.headers.pop("X-Cutpoint-Token")
    res = client.post(f"{URL}?t={token}", json={"key": FRAMES})
    assert res.status_code == 403
