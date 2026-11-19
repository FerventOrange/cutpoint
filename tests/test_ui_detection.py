"""Describe the footage, look closer with vision, and the style button of ``cutpoint ui``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")
import importlib.util

from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import patch_ui, set_config, write
from cutpoint import cache, cli, extras, ui
from ui_helpers import (
    CHAT,
    X3_CLIPS,
    JobSpy,
    _after,
    _cli_args,
    cli_project,
    progress_record,
    resume_job,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


# ------------------------------------------------ Describe the footage / Look closer with vision (#486)
def refine_project(tmp_path: Path, style: str = "action", **hl_over: object) -> Path:
    """A command-line project analysed in ``style`` (extra ``highlights.json`` keys)."""
    return cli_project(tmp_path, style=style, **hl_over)[0]


@pytest.mark.parametrize("body", [{"action": "analyze"}, {"action": "edit", "kind": "highlights"}])
def test_a_job_of_a_deleted_gaming_style_starts_the_proxy_for_the_chat_model(
    home: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict
) -> None:
    """The dialogue step of the base (gaming) needs the subscription proxy; the deleted name
    would not tell it, the ``--style`` the job runs with does."""
    set_config(home, **CHAT)
    deleted_style_project(tmp_path, "gaming")
    res, seen = refine_job(client, monkeypatch, body)
    assert res.status_code == 200, res.text
    assert _after(_cli_args(seen[0]["args"]), "--style") == "gaming"
    assert seen[0]["before"] is not None


def refine_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, body: dict, *, lite: str | None = None
) -> tuple[object, list[dict[str, object]]]:
    """POST ``body`` as a job of ``cli``, ``Jobs.start`` recording what it was given."""
    seen = JobSpy(monkeypatch).calls
    patch_ui(monkeypatch, "events_unavailable", lambda: lite)
    return client.post("/api/projects/cli/jobs", json=body), seen


def test_describe_the_footage_runs_crashes_with_the_context(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    refine_project(tmp_path)
    text = "  enduro race, mud on the lens  "
    res, seen = refine_job(client, monkeypatch, {"action": "crashes", "context": text})
    assert res.status_code == 200, res.text
    args = _cli_args(seen[0]["args"])  # parses with the CLI
    assert args == [
        "crashes",
        str(ui.projects_dir() / "cli"),
        "--context=enduro race, mud on the lens",
    ]
    parsed = cli.build_parser().parse_args(args)
    assert (parsed.context, parsed.reanalyze) == ("enduro race, mud on the lens", True)
    assert seen[0]["action"] == "crashes" and seen[0]["cwd"] == ui.projects_dir()


@pytest.mark.parametrize(
    ("body", "style", "message"),
    [
        ({"context": ""}, "action", "describe the footage first"),
        ({"context": "   "}, "action", "describe the footage first"),
        ({"context": 5}, "action", "describe the footage first"),
        ({}, "action", "describe the footage first"),
        ({"context": "x" * 501}, "action", "at most 500"),
        ({"context": "mud"}, "funny", "does not run for this footage"),
    ],
    ids=["empty", "blank", "number", "missing", "long", "no-crash-step"],
)
def test_describe_the_footage_is_refused_when_it_cannot_apply(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    style: str,
    message: str,
) -> None:
    refine_project(tmp_path, style)
    res, seen = refine_job(client, monkeypatch, {"action": "crashes", **body})
    assert res.status_code == 400 and message in res.json()["detail"]
    assert seen == []


@pytest.mark.parametrize("action", ["crashes", "events"])
def test_refining_is_refused_before_the_first_analysis_for_360_and_in_lite(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str
) -> None:
    body = {"action": action, "context": "mud"}
    p = ui.projects_dir() / "cli"
    write(p / "ui" / "meta.json", json.dumps({"style": "action", "source": str(tmp_path)}))
    res, seen = refine_job(client, monkeypatch, body)
    assert res.status_code == 400 and "analyze the footage first" in res.json()["detail"]

    cache.save_json(p / "highlights.json", {"style": "action", "settings": {}})
    cache.save_json(p / "ui" / "meta.json", {**ui.read_meta(p), "kind": "360"})
    res, _ = refine_job(client, monkeypatch, body)
    assert res.status_code == 400 and "360" in res.json()["detail"]

    cache.save_json(p / "ui" / "meta.json", {**ui.read_meta(p), "kind": "video"})
    res, _ = refine_job(client, monkeypatch, body, lite="x is not available in Cutpoint lite")
    assert res.status_code == 409 and "Cutpoint lite" in res.json()["detail"]
    assert seen == []


def test_look_closer_with_vision_runs_events_on_the_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    refine_project(tmp_path, "funny")  # any style: events has a VLM layer for all of them
    res, seen = refine_job(client, monkeypatch, {"action": "events"})
    assert res.status_code == 200, res.text
    args = _cli_args(seen[0]["args"])
    assert args == ["events", str(ui.projects_dir() / "cli")]
    assert seen[0]["action"] == "events" and seen[0]["on_result"] is not None


@pytest.mark.parametrize("action", ["crashes", "events"])
def test_the_vision_consent_warnings_reach_the_job_as_notes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str
) -> None:
    refine_project(tmp_path)
    _, seen = refine_job(client, monkeypatch, {"action": action, "context": "mud"})
    job = ui.Job("cli", action)
    job.add_output("12:00:00 WARNING already in the log\n")
    warnings = [
        {"code": "vision_consent_needed", "message": "frames would go to host:1; skipping vision"},
        {"code": "vision_unavailable", "message": "no vision model"},
        {"code": "vlm_unavailable", "message": "no VLM"},
        {"code": "context_not_applied", "message": "the Ollama model is unavailable"},
        {"code": "vision_consent_needed", "message": "already in the log"},
        {"code": "highlights_not_updated", "message": "a source could not be found"},
    ]
    job.result = {"warnings": warnings}
    seen[0]["on_result"](job)
    added = list(job.logs)[1:]
    assert [ln.split(" WARNING ", 1)[1] for ln in added] == [w["message"] for w in warnings[:4]]
    assert sum("already in the log" in ln for ln in job.logs) == 1  # the CLI's own line only


def test_a_refining_job_waits_for_the_subscription_proxy_of_the_vision_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, home: Path
) -> None:
    refine_project(tmp_path)
    _, seen = refine_job(client, monkeypatch, {"action": "events"})
    assert seen[0]["before"] is None
    set_config(home, vision_assistant={"provider": "codex"})
    _, seen = refine_job(client, monkeypatch, {"action": "events"})
    assert seen[0]["before"] is not None


@pytest.mark.parametrize(
    ("style", "hl_over", "crashes_json", "want"),
    [
        ("action", {}, False, True),
        ("sport", {}, False, True),
        ("action-cam", {}, False, True),
        ("funny", {}, False, False),
        ("funny", {}, True, True),  # sources with telemetry: the analysis wrote crashes.json
        ("funny", {"detection": {"category": "general"}}, False, True),  # --style auto
        ("funny", {"detection": {"category": "action"}}, False, True),  # --style auto
        ("funny", {"detection": {"category": "general", "fallback": True}}, False, False),
        ("funny", {"detection": {"category": "talk"}}, False, False),
    ],
)
def test_the_project_says_whether_the_crash_step_applies(
    client: TestClient, tmp_path: Path, style: str, hl_over: dict, crashes_json: bool, want: bool
) -> None:
    p = refine_project(tmp_path, style, **hl_over)
    if crashes_json:
        cache.save_json(p / "crashes.json", {"crashes": []})
    assert client.get("/api/projects/cli").json()["refine"]["crash_step"] is want


@pytest.mark.parametrize(
    ("meta", "want"),
    [
        ({"style": "action"}, True),
        ({"style": "sport"}, True),
        ({"style": "action-cam"}, True),
        ({"style": "auto"}, True),
        ({}, True),  # no style: the footage decides
        ({"style": "funny"}, False),
        ({"style": "action", "kind": "360"}, False),
    ],
)
def test_before_the_first_analysis_the_chosen_style_says_whether_the_crash_step_applies(
    client: TestClient, tmp_path: Path, meta: dict, want: bool
) -> None:
    p = ui.projects_dir() / "cli"
    write(p / "ui" / "meta.json", json.dumps({"source": str(tmp_path), **meta}))
    view = client.get("/api/projects/cli").json()
    assert view["highlights"] is None and view["refine"]["crash_step"] is want


@pytest.mark.parametrize(
    ("command", "proxy"), [("events", True), ("crashes", True), ("clips", False)]
)
def test_resuming_a_refining_run_starts_the_proxy_of_a_subscription_vision_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    home: Path,
    command: str,
    proxy: bool,
) -> None:
    p, _ = cli_project(tmp_path)
    progress_record(p, command=command, argv=[command, str(p)])
    set_config(home, vision_assistant={"provider": "codex"})
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    assert (seen[0]["before"] is not None) is proxy


def test_the_project_says_the_description_it_has_and_whether_events_run_here(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = refine_project(tmp_path)
    patch_ui(monkeypatch, "events_unavailable", lambda: None)
    view = client.get("/api/projects/cli").json()["refine"]
    assert (view["context"], view["unavailable"]) == (None, None)
    cache.save_json(p / "activity.json", {"request": {"activity": None, "context": "mud"}})
    patch_ui(monkeypatch, "events_unavailable", lambda: "torch is not installed")
    view = client.get("/api/projects/cli").json()["refine"]
    assert (view["context"], view["unavailable"]) == ("mud", "torch is not installed")


def test_the_lite_message_names_the_missing_module_and_how_to_add_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extras, "edition", lambda: "full")
    assert ui.events_unavailable() is None
    monkeypatch.setattr(extras, "edition", lambda: "lite")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    message = ui.events_unavailable()
    assert message is not None and "not available in Cutpoint lite" in message


# ---------- the header's style button (#897) ----------


def detection(
    cat: str = "gameplay",
    *,
    ambiguous: bool = False,
    style: str = "gaming",
    share: float = 1.0,
    sources: list[dict[str, Any]] | None = None,
    suggest: str | None = None,
) -> dict[str, Any]:
    """A highlights.json ``detection`` record (category.Detection.record) with invented sources."""
    src = {
        "path": "/v/a.mp4",
        "duration": 60.0,
        "category": cat,
        "ambiguous": ambiguous,
        "reason": "static HUD 2.0%",
        "gameplay": 0.9,
        "speech": 0.2,
        "decided_by": None if ambiguous else "static HUD",
    }
    return {
        "version": 3,
        "category": cat,
        "emphasis": "hype",
        "style": style,
        "reason": "static HUD 2.0%",
        "ambiguous": ambiguous,
        "scores": {cat: share},
        "fallback": style if ambiguous else None,
        "suggest": suggest,
        "decided_by": [] if ambiguous else ["static HUD"],
        "sources": sources or [src],
    }


def auto_project(
    tmp_path: Path, det: dict[str, Any] | None, meta: dict[str, Any] | None = None
) -> Path:
    """The command-line project ``cli`` analyzed with ``--style auto`` (``det`` None: with the
    style ``meta`` names, no detection), with ``meta`` as its ui/meta.json (None: none)."""
    if det is None:
        return cli_project(tmp_path, meta=meta, style=(meta or {}).get("style", "gaming"))[0]
    auto = {"style_requested": "auto"}
    return cli_project(tmp_path, settings=auto, meta=meta, style=det["style"], detection=det)[0]


def test_the_detected_view_of_a_sure_auto_project(client: TestClient, tmp_path: Path) -> None:
    auto_project(tmp_path, detection(), {"style": "auto"})
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["auto"], d["chosen"], d["style"]) == (True, "auto", "gaming")
    assert (d["category"], d["label"], d["reason"]) == ("gameplay", "Gameplay", "static HUD 2.0%")
    assert (d["ambiguous"], d["fallback_label"], d["stale"], d["sources"]) == (
        False,
        None,
        False,
        [],
    )
    # the categories with their style, from styles.AUTO_STYLES (not spelled out in the page)
    assert [(c["label"], c["style"], c["main"]) for c in d["choices"]] == [
        ("Gameplay", "gaming", True),
        ("Action", "action", True),
        ("Action cam", "action-cam", False),  # no camera telemetry: under Other styles
        ("Talk", "commentary", True),
    ]


def test_the_detected_view_of_action_footage(client: TestClient, tmp_path: Path) -> None:
    auto_project(tmp_path, detection("action", style="action"), {"style": "auto"})
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["category"], d["label"], d["style"]) == ("action", "Action", "action")
    labels = [c["label"] for c in d["choices"]]
    assert labels.count("Action") == 1 and "General" not in labels
    assert len({c["category"] for c in d["choices"]}) == len(d["choices"])
    assert len({c["style"] for c in d["choices"] if c["main"]}) == len(
        [c for c in d["choices"] if c["main"]]
    )  # no duplicate data-style items in the menu


def test_an_old_version_3_detection_of_general_footage_still_shows(
    client: TestClient, tmp_path: Path
) -> None:
    det = {**detection("general", style="action"), "version": 3}
    auto_project(tmp_path, det, {"style": "auto"})
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["category"], d["label"], d["stale"]) == ("general", "General", False)
    assert "General" not in [c["label"] for c in d["choices"]]


@pytest.mark.parametrize(
    ("cat", "style", "fallback"),
    [("gameplay", "gaming", "Gameplay"), ("talk", "action", "Action")],
    ids=["leans-gameplay", "leans-talk"],
)
def test_an_unsure_detection_names_the_category_whose_style_it_used(
    client: TestClient, tmp_path: Path, cat: str, style: str, fallback: str
) -> None:
    auto_project(tmp_path, detection(cat, ambiguous=True, style=style), {"style": "auto"})
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["category"], d["ambiguous"], d["fallback_label"], d["used_style"]) == (
        cat,
        True,
        fallback,
        style,
    )


@pytest.mark.parametrize(("share", "named"), [(0.85, True), (0.9, False)], ids=["15pct", "10pct"])
def test_the_detected_view_lists_the_sources_that_disagree_above_the_share(
    client: TestClient, tmp_path: Path, share: float, named: bool
) -> None:
    srcs = [
        {"path": "/v/a.mp4", "duration": 85.0, "category": "gameplay", "ambiguous": False,
         "reason": "static HUD 2.0%", "decided_by": "static HUD"},
        {"path": "/v/b.mp4", "duration": 15.0, "category": "talk", "ambiguous": True,
         "reason": "speech 50% of 30 min", "decided_by": None},
    ]  # fmt: skip
    auto_project(tmp_path, detection(share=share, sources=srcs), {"style": "auto"})
    d = client.get("/api/projects/cli").json()["detected"]
    want = [
        {"name": "b.mp4", "path": "/v/b.mp4", "category": "talk", "label": "Talk",
         "reason": "speech 50% of 30 min", "ambiguous": True}
    ]  # fmt: skip
    assert d["sources"] == (want if named else [])


def test_action_cam_is_a_main_choice_with_camera_telemetry_or_when_chosen(
    client: TestClient, tmp_path: Path
) -> None:
    det = detection("action-cam", style="action-cam")
    det["sources"][0]["decided_by"] = "telemetry"
    p = auto_project(tmp_path, det, {"style": "auto"})

    def main() -> dict[str, bool]:
        view = client.get("/api/projects/cli").json()["detected"]
        return {c["style"]: c["main"] for c in view["choices"]}

    assert main()["action-cam"] is True
    write(p / "ui" / "meta.json", json.dumps({"style": "action-cam"}))
    hl = cache.load_json(p / "highlights.json")
    hl.pop("detection")
    hl["settings"].pop("style_requested")
    cache.save_json(p / "highlights.json", hl)
    assert main()["action-cam"] is True
    write(p / "ui" / "meta.json", json.dumps({"style": "sport"}))
    hl["style"] = "sport"
    cache.save_json(p / "highlights.json", hl)
    assert main()["action-cam"] is False


def test_a_project_in_a_chosen_style_has_no_detection(client: TestClient, tmp_path: Path) -> None:
    auto_project(tmp_path, None, {"style": "commentary"})
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["auto"], d["chosen"], d["style"], d["category"], d["stale"]) == (
        False,
        "commentary",
        "commentary",
        None,
        False,
    )


def test_after_leaving_auto_the_view_shows_the_saved_copy_as_stale(
    client: TestClient, tmp_path: Path
) -> None:
    """The copy in meta.json is shown once the user chose a style, also while the analysis in
    it runs (highlights.json still holds the Auto run's detection)."""
    old = detection()
    p = auto_project(tmp_path, old, {"style": "commentary", "detection": old})
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["auto"], d["chosen"], d["category"], d["stale"]) == (
        False,
        "commentary",
        "gameplay",
        True,
    )
    hl = cache.load_json(p / "highlights.json")  # the analysis in commentary finished
    hl.pop("detection")
    hl["settings"].pop("style_requested")
    hl["style"] = "commentary"
    cache.save_json(p / "highlights.json", hl)
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["style"], d["category"], d["stale"]) == ("commentary", "gameplay", True)


def test_an_action_cam_project_that_fell_back_shows_the_style_that_ran() -> None:
    """#939: action-cam on footage without telemetry ran as action; the page shows action
    (a later switch to another style shows that one)."""
    hl = {
        "style": "action",
        "settings": {
            "style_requested": "action-cam",
            "style_fallback": {"from": "action-cam", "reason": "no telemetry"},
        },
    }
    assert ui.ProjectStyle.of({"style": "action-cam"}, hl).resolved == "action"
    assert ui.ProjectStyle.of({"style": "gaming"}, hl).resolved == "gaming"
    cam = {"style": "action-cam"}
    assert ui.ProjectStyle.of(cam, {"style": "action-cam"}).resolved == "action-cam"
    d = ui.detected_view({"style": "action-cam"}, hl)
    assert d is not None and (d["chosen"], d["style"]) == ("action-cam", "action")


def test_a_command_line_project_that_fell_back_analyzes_again_with_its_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#939: Analyze again of an action-cam project that ran as action asks for action-cam
    again (as reanalyze does), so the request stays recorded."""
    p = auto_project(tmp_path, None, None)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "action"
    hl["settings"].update(
        style_requested="action-cam",
        style_fallback={"from": "action-cam", "reason": "no telemetry"},
    )
    cache.save_json(p / "highlights.json", hl)
    captured = JobSpy(monkeypatch, check=False).args
    assert client.post("/api/projects/cli/jobs", json={"action": "analyze"}).status_code == 200
    assert _after(_cli_args(captured[0]), "--style") == "action-cam"


def test_no_detected_view_for_360_projects_or_before_the_first_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    assert ui.detected_view({"kind": "360", "style": "action-cam"}, {"style": "action-cam"}) is None
    assert ui.detected_view({"style": "auto"}, None) is None
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    pid = client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["detected"] is None


def test_a_command_line_auto_project_analyzes_again_with_style_auto(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without meta.json the recorded ``style_requested`` says the analysis was Auto: Analyze
    again keeps --style auto (not the style auto picked)."""
    auto_project(tmp_path, detection(), None)
    assert client.get("/api/projects/cli").json()["detected"]["chosen"] == "auto"
    captured = JobSpy(monkeypatch, check=False).args
    assert client.post("/api/projects/cli/jobs", json={"action": "analyze"}).status_code == 200
    assert _after(_cli_args(captured[0]), "--style") == "auto"


def cli_reranked_project(tmp_path: Path, meta: dict[str, Any] | None) -> tuple[Path, dict]:
    """The Auto-analyzed (gameplay) command-line project ``cli`` after ``cutpoint rerank --style
    funny``: highlights.json is in funny and records the re-rank and keeps the detection; the
    command line leaves ui/meta.json (``meta``) as it was."""
    det = detection()
    settings = {
        "style_requested": "funny",
        "reranked": {"from": "gaming", "requested": "auto"},
        "rerank_backup": "0123456789abcdef",
    }
    p, _ = cli_project(tmp_path, settings=settings, meta=meta, style="funny", detection=det)
    return p, det


def analyzed_again(p: Path) -> None:
    """The analysis Analyze again ran finished: highlights.json is in its style and records no
    re-rank (as ``cutpoint analyze --style NAME`` writes it)."""
    hl = cache.load_json(p / "highlights.json")
    for key in ("reranked", "rerank_backup"):
        hl["settings"].pop(key, None)
    cache.save_json(p / "highlights.json", hl)


def test_a_reranked_analysis_runs_on_its_style_over_the_chosen_auto() -> None:
    """#902: meta.json still says auto after a command-line re-rank; the analysis runs on funny."""
    hl = {"style": "funny", "settings": {"reranked": {"from": "gaming", "requested": "auto"}}}
    assert ui.ProjectStyle.of({"style": "auto"}, hl).chosen == "funny"
    assert ui.ProjectStyle.of({}, hl).chosen == "funny"
    plain = {"style": "funny", "settings": {"style_requested": "auto"}}
    assert ui.ProjectStyle.of({"style": "auto"}, plain).chosen == "auto"  # no re-rank: unchanged


def test_a_reranked_project_shows_its_style_and_the_detection_as_stale(
    client: TestClient, tmp_path: Path
) -> None:
    cli_reranked_project(tmp_path, {"style": "auto"})
    got = client.get("/api/projects/cli").json()
    assert got["style"] == "funny"
    d = got["detected"]
    assert (d["chosen"], d["style"], d["category"], d["stale"]) == (
        "funny",
        "funny",
        "gameplay",
        True,
    )
    assert got["ranking_key"] and "rerank_backup" not in got


def test_a_reranked_project_is_not_gaming_after_a_rerank_away_from_gaming(
    client: TestClient, tmp_path: Path
) -> None:
    """The Color default and the crash default follow the style the ranking is in."""
    p, _ = cli_reranked_project(tmp_path, {"style": "auto"})
    got = client.get("/api/projects/cli").json()
    assert got["gaming"] is False
    assert got["must_include_crash_default"] == ui._style_crash("funny", None)
    hl = cache.load_json(p / "highlights.json")  # the same project ranked in gaming again
    hl["style"] = "gaming"
    cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["gaming"] is True


def test_analyze_again_after_a_command_line_rerank_keeps_its_style(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Analyze again must not send --style auto back (it would undo the re-rank)."""
    cli_reranked_project(tmp_path, {"style": "auto"})
    res, captured = style_job(client, monkeypatch, {})
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "funny"


@pytest.mark.parametrize("action", ["analyze", "edit"])
def test_a_reranked_style_is_kept_after_the_next_analysis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str
) -> None:
    """The job saves the re-ranked style (and the detection) in meta.json: the analysis it runs
    records no re-rank, so the second Analyze again still sends funny, not meta's auto."""
    p, det = cli_reranked_project(tmp_path, {"style": "auto"})
    if action == "analyze":
        res, captured = style_job(client, monkeypatch, {})
    else:
        res, captured = edit_job(client, monkeypatch, {"kind": "highlights"})
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "funny"
    meta = cache.load_json(p / "ui" / "meta.json")
    assert (meta["style"], meta["detection"]) == ("funny", det)
    analyzed_again(p)
    res, captured = style_job(client, monkeypatch, {})
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "funny"
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["chosen"], d["category"], d["stale"]) == ("funny", "gameplay", True)


def test_the_header_pick_wins_over_a_rerank_until_the_next_rerank(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The user's last pick wins even when its analysis failed (highlights.json is still the
    re-ranked one); re-ranking the project after that pick wins again."""
    p, _ = cli_reranked_project(tmp_path, {"style": "auto"})
    res, _ = style_job(client, monkeypatch, {"style": "action"})
    assert res.status_code == 200, res.text  # the job then fails: highlights.json stays
    assert cache.load_json(p / "ui" / "meta.json")["style"] == "action"
    assert client.get("/api/projects/cli").json()["detected"]["chosen"] == "action"
    res, captured = style_job(client, monkeypatch, {})
    assert _after(_cli_args(captured[0]), "--style") == "action"
    hl = cache.load_json(p / "highlights.json")  # `cutpoint rerank --style funny` again
    hl["settings"]["rerank_backup"] = "fedcba9876543210"
    cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["detected"]["chosen"] == "funny"
    res, captured = style_job(client, monkeypatch, {})
    assert _after(_cli_args(captured[0]), "--style") == "funny"


def test_an_older_meta_without_the_rerank_mark_lets_the_rerank_win() -> None:
    hl = {
        "style": "funny",
        "settings": {"reranked": {"from": "gaming"}, "rerank_backup": "0123456789abcdef"},
    }
    assert ui.ProjectStyle.of({"style": "action"}, hl).chosen == "funny"
    assert ui.ProjectStyle.of({"style": "action", "style_for": None}, hl).chosen == "funny"
    marked = {"style": "action", "style_for": "0123456789abcdef"}
    assert ui.ProjectStyle.of(marked, hl).chosen == "action"


def deleted_style_project(tmp_path: Path, base: str | None) -> Path:
    """The command-line project ``cli`` analyzed in the user style ``gone`` (deleted since), which
    ui/meta.json also names; ``base``: the built-in style highlights.json records for it."""
    recorded = {} if base is None else {"style_base": base}
    return cli_project(tmp_path, meta={"style": "gone"}, style="gone", **recorded)[0]


def test_analyze_again_of_a_deleted_user_style_runs_in_its_base_and_keeps_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The job passes the recorded base, not the deleted name, and the project's chosen style
    becomes the base (so the options panel shows it, not an unknown entry)."""
    p = deleted_style_project(tmp_path, "action")
    res, captured = style_job(client, monkeypatch, {})
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "action"
    assert cache.load_json(p / "ui" / "meta.json")["style"] == "action"


def test_analyze_again_of_a_deleted_user_style_without_a_base_is_a_clear_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = deleted_style_project(tmp_path, None)
    res, captured = style_job(client, monkeypatch, {})
    assert res.status_code == 400 and captured == []
    assert res.json()["detail"] == (
        "The style 'gone' of this analysis is not in the style library (deleted?); "
        "pick another style"
    )
    assert cache.load_json(p / "ui" / "meta.json")["style"] == "gone"
    # choosing another style is still possible
    res, captured = style_job(client, monkeypatch, {"style": "action"})
    assert res.status_code == 200 and _after(_cli_args(captured[0]), "--style") == "action"


def edit_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> tuple[Any, list[list[str]]]:
    """Post an edit (preset) job to the project ``cli``; the job starts and its check runs."""
    captured = JobSpy(monkeypatch).args
    return client.post("/api/projects/cli/jobs", json={"action": "edit", **body}), captured


def test_a_preset_of_a_deleted_user_style_runs_in_its_base_and_keeps_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """edit records the base in highlights.json, so the project's chosen style becomes it too
    (or the next Analyze again would not find the style it recorded)."""
    p = deleted_style_project(tmp_path, "action")
    res, captured = edit_job(client, monkeypatch, {"kind": "highlights"})
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "action"
    assert cache.load_json(p / "ui" / "meta.json")["style"] == "action"


def test_a_preset_of_a_deleted_user_style_without_a_base_is_a_clear_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deleted_style_project(tmp_path, None)
    res, captured = edit_job(client, monkeypatch, {"kind": "highlights"})
    assert res.status_code == 400 and captured == []
    assert "pick another style" in res.json()["detail"]


def test_a_preset_of_a_deleted_user_style_ignores_the_crash_value_it_had(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The deleted style's own --must-include-crash is gone with it: the preset runs with the
    base's, not refused for a value it cannot repeat."""
    p = deleted_style_project(tmp_path, "action")
    hl = cache.load_json(p / "highlights.json")
    hl["settings"]["must_include_crash"] = 0.95
    cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["preset_problem"] is None
    res, _ = edit_job(client, monkeypatch, {"kind": "highlights"})
    assert res.status_code == 200, res.text


def test_360_analyze_of_a_deleted_user_style_runs_in_its_base_and_keeps_it(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    pid = client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    p = ui.project_dir(pid)
    meta = cache.load_json(p / "ui" / "meta.json")
    cache.save_json(p / "ui" / "meta.json", meta | {"style": "gone"})
    cache.save_json(p / "highlights.json", {"style": "gone", "style_base": "action-cam"})
    patch_ui(monkeypatch, "cutpoint_cmd", lambda *a: ["cutpoint", *a])
    seen = JobSpy(monkeypatch).args
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert _after(seen[0], "--style") == "action-cam"
    assert cache.load_json(p / "ui" / "meta.json")["style"] == "action-cam"
    cache.save_json(p / "highlights.json", {"style": "gone"})  # an older file: no base
    cache.save_json(p / "ui" / "meta.json", meta | {"style": "gone"})
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 400 and len(seen) == 1
    assert "pick another style" in res.json()["detail"]


def style_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    *,
    starts: bool = True,
) -> tuple[Any, list[list[str]]]:
    """Post an analyze job with ``body``; the job starts (its check runs) when ``starts``."""
    refuse = None if starts else HTTPException(409, "another job is already running")
    captured = JobSpy(monkeypatch, refuse=refuse).args
    return client.post("/api/projects/cli/jobs", json={"action": "analyze", **body}), captured


def test_choosing_a_style_analyzes_in_it_and_keeps_the_detection_as_a_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    det = detection()
    p = auto_project(tmp_path, det, {"name": "Trip", "style": "auto"})
    res, captured = style_job(client, monkeypatch, {"style": "commentary"})
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "commentary"
    meta = ui.read_meta(p)
    assert (meta["name"], meta["style"], meta["detection"]) == ("Trip", "commentary", det)
    d = client.get("/api/projects/cli").json()["detected"]
    assert (d["chosen"], d["stale"], d["label"]) == ("commentary", True, "Gameplay")
    # nothing is deleted
    assert (p / "highlights.json").is_file()

    res, captured = style_job(client, monkeypatch, {"style": "auto"})  # back to Auto
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--style") == "auto"
    meta = ui.read_meta(p)
    assert (meta["style"], meta["detection"]) == ("auto", det)


def test_a_refused_style_job_leaves_the_style_as_it_was(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = auto_project(tmp_path, detection(), {"style": "auto"})
    before = (p / "ui" / "meta.json").read_bytes()
    res, _ = style_job(client, monkeypatch, {"style": "commentary"}, starts=False)
    assert res.status_code == 409
    assert (p / "ui" / "meta.json").read_bytes() == before
    # a run elsewhere holds the project: refused when the job would start
    patch_ui(monkeypatch, "project_busy", lambda *_a: True)
    res, _ = style_job(client, monkeypatch, {"style": "commentary"})
    assert res.status_code == 409 and res.json()["detail"] == ui.BUSY_TEXT
    assert (p / "ui" / "meta.json").read_bytes() == before


def test_a_style_job_that_cannot_start_puts_meta_back(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command fails to start after the check wrote the style: the old meta.json returns,
    and a command-line project gets none."""
    p = auto_project(tmp_path, detection(), None)

    def start(self, project_id, action, args, on_success, same_folder, *rest, **kw):
        same_folder()
        assert ui.read_meta(p)["style"] == "action"
        kw["on_refused"]()  # as Jobs.start does, under its lock
        raise HTTPException(500, "could not start job")

    monkeypatch.setattr(ui.Jobs, "start", start)
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", "style": "action"})
    assert res.status_code == 500
    assert not (p / "ui" / "meta.json").exists()


def test_a_real_start_failure_puts_meta_back_and_removes_the_ui_folder_it_made(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command cannot be spawned: Jobs.start undoes the style change under its lock, the
    old meta.json returns byte for byte and a command-line project keeps no empty ui folder."""

    def no_spawn(*_a, **_kw):
        raise OSError("no such program")

    patch_ui(monkeypatch, "spawn", no_spawn)
    p = auto_project(tmp_path, None, None)
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", "style": "action"})
    assert res.status_code == 500
    assert not (p / "ui").exists()

    q = ui.projects_dir() / "cli"
    write(q / "ui" / "meta.json", json.dumps({"style": "gaming"}))
    before = (q / "ui" / "meta.json").read_bytes()
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", "style": "funny"})
    assert res.status_code == 500
    assert (q / "ui" / "meta.json").read_bytes() == before


def test_a_command_line_project_gets_a_meta_with_the_chosen_style(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = auto_project(tmp_path, None, None)  # analyzed as gaming on the command line
    assert not (p / "ui").exists()
    res, captured = style_job(client, monkeypatch, {"style": "funny"})
    assert res.status_code == 200, res.text
    assert ui.read_meta(p) == {"style": "funny"}  # no Auto left: no copy
    assert _after(_cli_args(captured[0]), "--style") == "funny"


def test_an_unknown_style_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = auto_project(tmp_path, detection(), {"style": "auto"})
    res, captured = style_job(client, monkeypatch, {"style": "nope"})
    assert res.status_code == 400 and "unknown style" in res.json()["detail"]
    assert captured == [] and ui.read_meta(p)["style"] == "auto"


def test_the_style_of_a_360_project_cannot_be_changed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = ui.projects_dir() / "cli"
    write(
        p / "ui" / "meta.json",
        json.dumps({"kind": "360", "style": "action-cam", "source": str(tmp_path)}),
    )
    res, captured = style_job(client, monkeypatch, {"style": "action"})
    assert res.status_code == 400 and "360" in res.json()["detail"]
    assert captured == [] and ui.read_meta(p)["style"] == "action-cam"


def test_a_style_change_repeats_the_options_the_earlier_analysis_used(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The recorded options are read against the earlier analysis's style: action-cam records
    "scenes": false without --no-scenes, so the analysis in action detects scenes (#880), and
    its own crash default is not passed on as an explicit --must-include-crash."""
    # all built-in styles share one default, so give the earlier style its own to tell them apart
    patch_ui(monkeypatch, "_style_crash", lambda s, _hl=None: 0.3 if s == "action-cam" else 0.6)
    p = auto_project(tmp_path, None, {"style": "action-cam"})
    hl = cache.load_json(p / "highlights.json")
    hl["settings"].update(scenes=False, must_include_crash=0.3)
    cache.save_json(p / "highlights.json", hl)
    res, captured = style_job(client, monkeypatch, {"style": "action"})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert _after(args, "--style") == "action"
    assert "--no-scenes" not in args
    assert not any(a.startswith("--must-include-crash") for a in args)
    # a --no-scenes the earlier analysis did use is kept
    hl["style"] = "gaming"
    cache.save_json(p / "highlights.json", hl)
    write(p / "ui" / "meta.json", json.dumps({"style": "gaming"}))
    res, captured = style_job(client, monkeypatch, {"style": "commentary"})
    assert "--no-scenes" in _cli_args(captured[0])


def test_a_style_switch_that_did_not_finish_does_not_change_the_recorded_options(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """meta.json changes when the job starts, highlights.json only when it succeeds: the options
    are judged against the style highlights.json records, not the one meta.json now names."""
    p = auto_project(tmp_path, None, {"style": "action-cam"})
    hl = cache.load_json(p / "highlights.json")
    hl["settings"].update(scenes=False)
    cache.save_json(p / "highlights.json", hl)
    write(p / "ui" / "meta.json", json.dumps({"style": "gaming"}))  # a failed switch
    res, captured = style_job(client, monkeypatch, {})
    assert res.status_code == 200, res.text
    assert "--no-scenes" not in _cli_args(captured[0])
    assert client.get("/api/projects/cli").json()["analyze_options"]["no_scenes"] is False


def test_a_custom_styles_crash_value_survives_a_style_switch_that_did_not_finish(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_ui(monkeypatch, "_style_crash", lambda s, _hl=None: 0.8 if s == "mine" else 0.6)
    p = auto_project(tmp_path, None, {"style": "mine"})
    hl = cache.load_json(p / "highlights.json")
    hl["settings"].update(must_include_crash=0.8)
    cache.save_json(p / "highlights.json", hl)
    write(p / "ui" / "meta.json", json.dumps({"style": "action"}))  # a failed switch
    res, captured = style_job(client, monkeypatch, {})
    assert res.status_code == 200, res.text
    assert not any(a.startswith("--must-include-crash") for a in _cli_args(captured[0]))
    assert client.get("/api/projects/cli").json()["analyze_options"]["must_include_crash"] is None


def test_a_step_failed_warning_names_its_sources_in_the_job_log() -> None:
    job = ui.Job("p", "events")
    job.result = {
        "warnings": [
            {"code": "step_failed", "message": "reactions failed", "files": ["a.mp4", "b.mp4"]},
            {"code": "step_failed", "message": "keyframes failed", "files": []},
            {"code": "other", "message": "not shown", "files": ["c.mp4"]},
        ]
    }
    ui._note_warnings(job)
    assert any(line.endswith("reactions failed (a.mp4, b.mp4)") for line in job.logs)
    assert not any("not shown" in line for line in job.logs)
    assert any(line.endswith("keyframes failed") for line in job.logs)
