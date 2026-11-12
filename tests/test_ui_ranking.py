"""``cutpoint ui``'s ranking routes (#902): preview a project in another library style without a
write, apply it after a click, the refusals, and no torch in the UI process."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from cutpoint import analyze, cache, folders, rerank, runstate, ui
from cutpoint.styles import get_style
from test_rerank import lite_layers, make_project, read  # noqa: F401 (lite_layers: a fixture)

PORT = 8765
BASE = "/api/projects/p/ranking"


@pytest.fixture
def client(home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(ui, "cuda_probe", lambda: False)  # never a child process here
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


def snapshot(project: Path) -> dict[str, bytes]:
    return {
        str(f.relative_to(project)): f.read_bytes()
        for f in sorted(project.rglob("*"))
        if f.is_file()
    }


def preview(client: TestClient, style: str = "funny") -> Any:
    return client.post(f"{BASE}/preview", json={"style": style})


def apply(client: TestClient, base: str, style: str = "funny") -> Any:
    return client.post(f"{BASE}/apply", json={"style": style, "base": base})


def hold_lock(project: Path) -> None:
    """A run on another machine holds the project's run lock."""
    lock = folders.lock_path(project / runstate.LOCK_NAME)
    lock.write_text(json.dumps({"pid": 1, "host": "elsewhere", "token": "t"}), encoding="utf-8")


def no_paths(payload: Any, *roots: Path) -> None:
    text = json.dumps(payload)
    for root in roots:
        for form in (str(root), str(root).replace("\\", "/"), str(root).replace("\\", "\\\\")):
            assert form not in text


# --------------------------------------------------------------------------- preview
def test_a_preview_lists_both_rankings_and_writes_nothing(home, tmp_path, client):
    project = make_project(home, tmp_path)
    before = snapshot(project)
    res = preview(client)
    assert res.status_code == 200, res.text
    got = res.json()
    assert got["ready"] is True and (got["style"], got["from"]) == ("funny", "gaming")
    assert got["base"] == rerank.key((project / "highlights.json").read_bytes())
    want = rerank.rerank(project, get_style("funny"), dry_run=True).after
    assert 0 < len(got["top"]) <= ui.PREVIEW_TOP
    assert [r["cli_id"] for r in got["top"]] == [c["id"] for c in want["candidates"]][
        : len(got["top"])
    ]
    assert {"rank", "id", "cli_id", "file", "start", "end", "score", "label", "was"} <= set(
        got["top"][0]
    )
    assert got["top"][0]["file"] in {"s0.mp4", "s1.mp4"}
    assert got["before"] and "suggested_top" in got and "notes" in got
    no_paths(got, home, tmp_path)
    assert snapshot(project) == before  # not even a backup


@pytest.mark.parametrize("style", ["auto", "no-such-style", "", None])
def test_a_preview_or_apply_of_auto_or_an_unknown_style_is_400(home, tmp_path, client, style):
    project = make_project(home, tmp_path)
    before = snapshot(project)
    assert client.post(f"{BASE}/preview", json={"style": style}).status_code == 400
    assert client.post(f"{BASE}/apply", json={"style": style, "base": "x"}).status_code == 400
    assert snapshot(project) == before


def test_a_360_project_is_refused(home, tmp_path, client):
    project = make_project(home, tmp_path)
    (project / "pano.json").write_text("{}", encoding="utf-8")
    before = snapshot(project)
    assert preview(client).status_code == 409
    assert apply(client, "x").status_code == 409
    assert snapshot(project) == before


def test_a_preview_is_refused_while_a_run_holds_the_project(home, tmp_path, client):
    hold_lock(make_project(home, tmp_path))
    res = preview(client)
    assert res.status_code == 409 and res.json()["code"] == "folder_busy"


# --------------------------------------------------------------------------- apply
def test_apply_writes_the_ranking_the_backup_and_the_chosen_style(home, tmp_path, client):
    project = make_project(home, tmp_path)
    old = (project / "highlights.json").read_bytes()
    shown = preview(client).json()
    res = apply(client, shown["base"])
    assert res.status_code == 200, res.text
    got = res.json()
    assert (got["style"], got["from"], got["backup"]) == ("funny", "gaming", True)
    assert "new_ids" in got and "dropped_ids" in got and "seconds" in got
    no_paths(got, home, tmp_path)
    hl = read(project)
    assert hl["style"] == "funny"
    assert [c["id"] for c in hl["candidates"]][: len(shown["top"])] == [
        r["cli_id"] for r in shown["top"]
    ]
    assert (project / rerank.BACKUP).read_bytes() == old
    assert ui.read_meta(project)["style"] == "funny"
    assert got["key"] == rerank.key((project / "highlights.json").read_bytes())
    page = client.get("/api/projects/p").json()
    assert page["style"] == "funny" and page["ranking_key"] == got["key"]
    assert page["rerank_backup"] is True


def test_apply_leaving_auto_keeps_the_detection(home, tmp_path, client):
    detection = {"category": "gameplay", "emphasis": "hype", "style": "gaming", "sources": []}
    project = make_project(home, tmp_path, style_requested="auto")
    hl = read(project)
    hl["detection"] = detection
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    assert apply(client, preview(client).json()["base"]).status_code == 200
    meta = ui.read_meta(project)
    assert (meta["style"], meta["detection"]) == ("funny", detection)
    page = client.get("/api/projects/p").json()
    assert page["detected"]["stale"] is True and page["detected"]["category"] == "gameplay"


def test_apply_after_the_file_changed_is_409_stale_and_writes_nothing(home, tmp_path, client):
    project = make_project(home, tmp_path)
    base = preview(client).json()["base"]
    hl = read(project)
    hl["suggested_top"] = 99  # an analysis or another ranking ran meanwhile
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    before = snapshot(project)
    res = apply(client, base)
    assert res.status_code == 409 and res.json()["code"] == "stale"
    assert "preview it again" in res.json()["detail"]
    assert snapshot(project) == before
    assert not (project / rerank.BACKUP).exists()
    assert not (project / "ui" / "meta.json").exists()


def test_apply_when_a_cache_went_missing_is_409_needs_analyze(home, tmp_path, client):
    project = make_project(home, tmp_path)
    base = preview(client).json()["base"]
    for f in cache.source_dir(tmp_path / "footage" / "s0.mp4").iterdir():
        f.unlink()  # every cache of one source
    before = snapshot(project)
    res = apply(client, base)
    got = res.json()
    assert res.status_code == 409 and got["code"] == "needs_analyze", res.text
    assert got["missing"] and got["can_analyze"] is True
    assert {"feature", "detail", "file"} <= set(got["missing"][0])
    no_paths(got, home, tmp_path)
    assert snapshot(project) == before
    again = preview(client).json()  # the preview says the same
    assert again["ready"] is False and again["missing"] and again["can_analyze"] is True
    no_paths(again, home, tmp_path)


def test_apply_while_a_job_runs_on_the_project_is_409_folder_busy(home, tmp_path, client):
    project = make_project(home, tmp_path)
    base = preview(client).json()["base"]
    job = ui.Job("p", "analyze")
    client.app.state.jobs.jobs[job.id] = job
    before = snapshot(project)
    res = apply(client, base)
    assert res.status_code == 409 and res.json()["code"] == "folder_busy"
    assert snapshot(project) == before
    job.status = "done"
    assert apply(client, base).status_code == 200


def test_apply_while_the_lock_is_held_is_409_folder_busy(home, tmp_path, client):
    project = make_project(home, tmp_path)
    base = preview(client).json()["base"]
    hold_lock(project)
    res = apply(client, base)
    assert res.status_code == 409 and res.json()["code"] == "folder_busy"
    assert read(project)["style"] == "gaming"


def test_a_job_does_not_start_while_the_ranking_is_applied(home, tmp_path, client, monkeypatch):
    project = make_project(home, tmp_path)
    started: list[Any] = []

    def start(self, project_id, action, args, on_success, same_folder, *rest, **_kw):
        same_folder()  # the check Jobs.start runs under its lock: it refuses
        started.append(args)
        return SimpleNamespace(view=lambda: {"status": "running"})

    monkeypatch.setattr(ui.Jobs, "start", start)
    ui.RERANKING.add(project)
    try:
        res = client.post("/api/projects/p/jobs", json={"action": "analyze"})
    finally:
        ui.RERANKING.discard(project)
    assert res.status_code == 409 and started == []


def test_the_marker_is_released_when_the_ranking_is_refused(home, tmp_path, client):
    make_project(home, tmp_path)
    assert apply(client, "not-the-key").status_code == 409
    assert not ui.RERANKING


# --------------------------------------------------------------------------- no torch
def test_the_ui_process_never_imports_torch_to_rank(home, tmp_path, client, monkeypatch):
    project = make_project(home, tmp_path)
    monkeypatch.setattr(rerank, "_events_extra", lambda: True)  # the events extra "installed"
    monkeypatch.setattr(analyze, "EVENTS_MODULES", ())

    def boom() -> bool:
        raise AssertionError("torch's own CUDA check ran in the UI process")

    monkeypatch.setattr(analyze, "_cuda", boom)
    asked: list[int] = []
    monkeypatch.setattr(ui, "cuda_probe", lambda: asked.append(1) or False)
    had_torch = "torch" in sys.modules
    got = preview(client, "action")  # plans the visual layer: asks the probe
    assert got.status_code == 200, got.text
    assert asked, "the injected probe was not asked"
    base = preview(client).json()["base"]
    assert apply(client, base).status_code == 200
    assert ("torch" in sys.modules) == had_torch
    assert read(project)["style"] == "funny"
