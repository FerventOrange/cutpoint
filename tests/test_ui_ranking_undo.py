"""``cutpoint ui``'s Undo / Redo of the last ranking change (#902): the swap of highlights.json
and its backup, what hides it (a rewritten analysis, an edited backup, other sources), the
refusals, the all-or-nothing swap and no torch in the UI process."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from conftest import patch_ui
from cutpoint import analyze, rerank, ui
from cutpoint.styles import get_style
from test_rerank import lite_layers, make_project, read  # noqa: F401 (lite_layers: a fixture)
from test_ui_ranking import (  # noqa: F401 (torch_imports: a fixture)
    apply,
    hold_lock,
    preview,
    snapshot,
    torch_imports,
)
from test_untrusted_paths import flushed_before_replace, fsyncs  # noqa: F401 (a fixture)

URL = "/api/projects/p/ranking/undo"


@pytest.fixture
def no_cuda(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_ui(monkeypatch, "cuda_probe", lambda: False)


@pytest.fixture
def client(no_cuda: None, ui_client: TestClient) -> TestClient:
    """The shared ``ui_client``, made after the CUDA probe is stubbed."""
    return ui_client


def page(client: TestClient) -> dict[str, Any]:
    return client.get("/api/projects/p").json()


def swap(client: TestClient, base: str | None = None) -> Any:
    return client.post(
        URL, json={"base": base if base is not None else page(client)["ranking_key"]}
    )


def applied(client: TestClient) -> None:
    res = apply(client, preview(client).json()["base"])
    assert res.status_code == 200, res.text


def rewrite(project: Path, name: str, edit: Callable[[dict[str, Any]], object]) -> None:
    hl = json.loads((project / name).read_text(encoding="utf-8"))
    edit(hl)
    (project / name).write_text(json.dumps(hl), encoding="utf-8")


def relink(project: Path) -> None:
    """highlights.json linked to the backup as it is now and sealed again: only the backup's own
    change is left for the Undo to notice."""
    link = analyze.backup_digest((project / rerank.BACKUP).read_bytes())
    rewrite(
        project, "highlights.json", lambda hl: hl["settings"].__setitem__("rerank_backup", link)
    )
    rewrite(project, "highlights.json", rerank.sealed)


def ranking(project: Path) -> tuple[str, list[tuple[str, float, float]]]:
    hl = read(project)
    return hl["style"], [(c["id"], c["start"], c["end"]) for c in hl["candidates"]]


def temps(project: Path) -> list[Path]:
    return [f for f in project.iterdir() if f.name.endswith(".tmp")]


# --------------------------------------------------------------------------- the round trip
def test_undo_redo_round_trip_restores_ids_candidates_and_style(home, tmp_path, client):
    project = make_project(home, tmp_path)
    assert page(client)["ranking_undo"] is None  # nothing was re-ranked yet
    first = ranking(project)
    applied(client)
    second = ranking(project)
    assert second[0] == "funny" and second != first
    assert page(client)["ranking_undo"] == {"action": "undo", "style": "gaming"}

    res = swap(client)
    assert res.status_code == 200, res.text
    got = res.json()
    assert got["action"] == "undo"
    assert ranking(project) == first
    assert ui.read_meta(project)["style"] == "gaming"
    assert page(client)["style"] == "gaming"
    assert page(client)["ranking_undo"] == {"action": "redo", "style": "funny"}

    got = swap(client).json()
    assert got["action"] == "redo"
    assert ranking(project) == second
    assert ui.read_meta(project)["style"] == "funny"
    assert page(client)["ranking_undo"] == {"action": "undo", "style": "gaming"}

    assert swap(client).json()["action"] == "undo"
    assert ranking(project) == first
    assert not temps(project)


def test_undo_back_to_an_auto_analysis_shows_auto_and_redo_keeps_the_detection(
    home, tmp_path, client
):
    project = make_project(home, tmp_path)
    detection = {"category": "gaming", "decided_by": "audio", "confidence": 0.9}

    def as_auto(hl: dict[str, Any]) -> None:
        hl["settings"]["style_requested"] = "auto"
        hl["detection"] = detection

    rewrite(project, "highlights.json", as_auto)
    applied(client)
    assert read(project)["style"] == "funny"
    assert read(project).get("detection") == detection

    assert swap(client).json()["action"] == "undo"
    assert ui.read_meta(project)["style"] == "auto"
    assert page(client)["style"] == "auto"

    assert swap(client).json()["action"] == "redo"
    assert ui.read_meta(project)["style"] == "funny"
    assert ui.read_meta(project)["detection"] == detection


def test_a_command_line_rerank_after_an_undo_still_restores_the_ids(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    funny = {(s, e): i for i, s, e in ranking(project)[1]}
    assert swap(client).status_code == 200
    assert read(project)["style"] == "gaming"
    rerank.rerank(project, get_style("funny"))  # as `cutpoint rerank --style funny`
    again = {(s, e): i for i, s, e in ranking(project)[1]}
    assert read(project)["style"] == "funny"
    shared = set(funny) & set(again)
    assert shared
    assert {k: funny[k] for k in shared} == {k: again[k] for k in shared}


def test_last_candidate_id_never_goes_down_across_undo_and_redo(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    funny_last = read(project)["settings"]["last_candidate_id"]
    assert funny_last > 1
    # the ranking in the backup knew fewer ids than the one in use gave out
    rewrite(project, rerank.BACKUP, lambda hl: hl["settings"].__setitem__("last_candidate_id", 1))
    relink(project)
    for _ in range(3):
        assert swap(client).status_code == 200
        assert read(project)["settings"]["last_candidate_id"] == funny_last


# --------------------------------------------------------------------------- stale / hidden
def test_a_wrong_base_is_409_stale_and_writes_nothing(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    before = snapshot(project)
    res = swap(client, "not-the-key")
    assert res.status_code == 409 and res.json()["code"] == "stale"
    assert snapshot(project) == before
    assert not ui.RERANKING


@pytest.mark.parametrize(
    "edit",
    [
        lambda hl: hl.__setitem__("suggested_top", 99),  # a hand edit, or `events`
        lambda hl: hl["settings"].pop(rerank.UNDO),  # an analysis records no seal
    ],
    ids=["hand-edited", "unsealed"],
)
def test_a_rewritten_highlights_file_hides_the_undo_and_the_route_refuses(
    home, tmp_path, client, edit
):
    project = make_project(home, tmp_path)
    applied(client)
    assert page(client)["ranking_undo"] is not None
    rewrite(project, "highlights.json", edit)
    assert page(client)["ranking_undo"] is None
    before = snapshot(project)
    res = swap(client)
    assert res.status_code == 409 and "code" not in res.json()
    assert snapshot(project) == before


def test_an_edited_backup_hides_the_undo(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    assert page(client)["ranking_undo"] is not None
    rewrite(project, rerank.BACKUP, lambda hl: hl.__setitem__("suggested_top", 99))
    assert page(client)["ranking_undo"] is None
    before = snapshot(project)
    assert swap(client).status_code == 409
    assert snapshot(project) == before


def test_a_backup_of_other_sources_hides_the_undo(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    other = str(tmp_path / "footage" / "elsewhere.mp4")
    rewrite(project, rerank.BACKUP, lambda hl: hl["sources"][0].__setitem__("path", other))
    relink(project)  # linked and sealed: only the sources differ
    assert page(client)["ranking_undo"] is None
    assert swap(client).status_code == 409


def test_the_view_follows_the_files_not_a_remembered_answer(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    assert page(client)["ranking_undo"] is not None
    assert page(client)["ranking_undo"] is not None  # remembered
    (project / rerank.BACKUP).unlink()
    assert page(client)["ranking_undo"] is None


# --------------------------------------------------------------------------- refusals
def test_a_project_without_a_backup_has_no_undo(home, tmp_path, client):
    project = make_project(home, tmp_path)
    before = snapshot(project)
    assert page(client)["ranking_undo"] is None
    res = swap(client)
    assert res.status_code == 409 and "code" not in res.json()
    assert snapshot(project) == before
    assert not (project / "ui").exists()  # no ui folder left behind for a swap it did not make


def test_the_swap_is_refused_while_the_lock_is_held(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    base = page(client)["ranking_key"]
    hold_lock(project)
    before = snapshot(project)
    res = swap(client, base)
    assert res.status_code == 409 and res.json()["code"] == "folder_busy"
    assert snapshot(project) == before
    assert not ui.RERANKING


def test_a_360_project_has_no_undo_and_is_refused(home, tmp_path, client):
    project = make_project(home, tmp_path)
    applied(client)
    base = page(client)["ranking_key"]
    (project / "pano.json").write_text("{}", encoding="utf-8")
    assert page(client)["ranking_undo"] is None
    before = snapshot(project)
    assert swap(client, base).status_code == 409
    assert snapshot(project) == before


def test_a_missing_base_is_400(home, tmp_path, client):
    make_project(home, tmp_path)
    assert client.post(URL, json={}).status_code == 400
    assert client.post(URL, json={"base": ""}).status_code == 400


# --------------------------------------------------------------------------- all or nothing
def test_a_failed_second_rename_leaves_both_files_and_no_temp(home, tmp_path, client, monkeypatch):
    project = make_project(home, tmp_path)
    applied(client)
    base = page(client)["ranking_key"]
    before = snapshot(project)
    real = Path.replace

    def flaky(self: Path, target: Any) -> Path:
        if Path(target).name == rerank.BACKUP:  # the second rename
            raise OSError("disk trouble")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    with pytest.raises(OSError, match="disk trouble"):
        ui.ranking_swap(project, base)
    monkeypatch.undo()
    assert snapshot(project) == before  # highlights.json put back, the backup untouched
    assert not temps(project)


# --------------------------------------------------------------------------- A639 / A667
LOGIN = "http://alice:s3c@ret@myhost:11434"


def no_login(project: Path) -> None:
    for name in ("highlights.json", rerank.BACKUP):
        raw = (project / name).read_bytes()
        assert b"alice" not in raw and b"s3c" not in raw, name


def read_backup_ranking(project: Path) -> tuple[str, list[tuple[str, float, float]]]:
    hl = json.loads((project / rerank.BACKUP).read_bytes())
    return hl["style"], [(c["id"], c["start"], c["end"]) for c in hl["candidates"]]


def test_undo_and_redo_keep_no_ollama_login_in_either_file(home, tmp_path, client):
    """A639: files written before #1119 kept the login in ``settings.ollama_url``, both the
    ranking in use and the backup the Undo brings back: neither file keeps it after a swap,
    and the swaps still go both ways."""
    project = make_project(home, tmp_path)
    applied(client)
    first, second = read_backup_ranking(project), ranking(project)
    for name in (rerank.BACKUP, "highlights.json"):
        rewrite(project, name, lambda hl: hl["settings"].__setitem__("ollama_url", LOGIN))
    relink(project)
    assert page(client)["ranking_undo"] == {"action": "undo", "style": "gaming"}

    assert swap(client).json()["action"] == "undo"
    no_login(project)
    assert ranking(project) == first
    for name in ("highlights.json", rerank.BACKUP):
        hl = json.loads((project / name).read_bytes())
        assert hl["settings"]["ollama_url"] == "http://myhost:11434"
    assert page(client)["ranking_undo"] == {"action": "redo", "style": "funny"}

    assert swap(client).json()["action"] == "redo"
    no_login(project)
    assert ranking(project) == second
    assert page(client)["ranking_undo"] == {"action": "undo", "style": "gaming"}
    assert not temps(project)


def test_undo_is_offered_after_applying_over_a_file_with_an_ollama_login(home, tmp_path, client):
    """A639: the apply drops the login an older file kept; the file it saves still matches its
    seal, so the Undo is offered and brings the earlier ranking back."""
    project = make_project(home, tmp_path, ollama_url="http://user:pass@host:11434")
    first = ranking(project)
    applied(client)
    assert page(client)["ranking_undo"] == {"action": "undo", "style": "gaming"}
    assert swap(client).json()["action"] == "undo"
    assert ranking(project) == first
    for name in ("highlights.json", rerank.BACKUP):
        assert b"user:pass" not in (project / name).read_bytes(), name


def test_the_swap_flushes_both_files_before_their_rename(
    home,
    tmp_path,
    client,
    fsyncs,  # noqa: F811
):
    """A667: highlights.json and its backup are flushed to disk before they are renamed into
    place, and on POSIX the project folder after the renames."""
    project = make_project(home, tmp_path)
    applied(client)
    fsyncs.clear()
    assert swap(client).status_code == 200
    assert flushed_before_replace(fsyncs, "highlights.json")
    assert flushed_before_replace(fsyncs, rerank.BACKUP)
    if os.name == "posix":
        at = next(i for i, c in enumerate(fsyncs) if c[:2] == ("replace", rerank.BACKUP))
        assert ("fsync", project.stat().st_ino) in fsyncs[at + 1 :]


# --------------------------------------------------------------------------- no torch
def test_the_ui_process_never_imports_torch_to_undo(
    home,
    tmp_path,
    client,
    monkeypatch,
    torch_imports,  # noqa: F811
):
    project = make_project(home, tmp_path)
    applied(client)

    def boom() -> bool:
        raise AssertionError("torch's own CUDA check ran in the UI process")

    monkeypatch.setattr(analyze, "_cuda", boom)
    assert swap(client).status_code == 200
    assert swap(client).status_code == 200
    assert torch_imports == []
    assert read(project)["style"] == "funny"
