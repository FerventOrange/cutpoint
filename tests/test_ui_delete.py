"""Delete a project (Settings): the plan, what stays, the trash, locks and links."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("httpx")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import link_dir, link_file, write
from cutpoint import folders, projectdelete, projectfiles, runstate, ui
from cutpoint.ui.routes import jobs as jobs_routes

REAL_TRASH_BLOCKER = projectdelete.trash_blocker


@pytest.fixture(autouse=True)
def bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in Recycle Bin inside tmp_path: no test ever trashes a real file."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()

    def fake(paths: list[str]) -> None:
        for p in paths:
            shutil.move(p, bin_ / f"{len(list(bin_.iterdir()))}-{Path(p).name}")

    monkeypatch.setattr(projectdelete.send2trash, "send2trash", fake)
    # never probe the host's registry or container state
    monkeypatch.setattr(projectdelete, "trash_blocker", lambda items: None)
    return bin_


def manifest(folder: Path, clips: list[str]) -> None:
    entries = [{"file": c, "srt": Path(c).with_suffix(".srt").name} for c in clips]
    write(folder / "manifest.json", json.dumps({"generator": "cutpoint", "clips": entries}))


def make(name: str = "trip") -> Path:
    """A project with Cutpoint's result files, one clips folder and a few files of the user's."""
    p = ui.projects_dir() / name
    write(p / "ui" / "meta.json", json.dumps({"name": "My Trip"}))
    write(p / "ui" / "proxy" / "a.mp4", "x" * 1000)
    write(p / "highlights.json", "{}")
    write(p / "transcript.txt", "hello")
    write(p / "progress.json", "{}")
    write(p / "final.mp4", "v" * 500)
    write(p / "final.srt")
    write(p / "trip-final.mp4")
    write(p / "clips" / "001_a.mp4", "c" * 100)
    write(p / "clips" / "001_a.srt")
    write(p / "clips" / "002_b.mp4")
    manifest(p / "clips", ["001_a.mp4", "002_b.mp4"])
    stale = write(p / ".clips.cutpoint-lock")  # a lock file a killed run left (never written)
    os.utime(stale, (0, 0))
    write(p / ".clips_ui_20260101-101010.cutpoint-resume" / "resume.json")
    return p


def test_the_plan_names_the_size_and_what_stays(ui_client: TestClient) -> None:
    p = make()
    write(p / "notes.txt", "mine")
    write(p / "clips" / "hand-made.mp4")
    write(p / "staged" / "copy.mp4")
    got = ui_client.get("/api/projects/trip/delete-plan").json()
    assert got["name"] == "My Trip"
    assert got["size"] >= 1600
    assert got["kept_count"] == 3
    assert sorted(got["kept"]) == ["clips/hand-made.mp4", "notes.txt", "staged/"]
    assert got["deleted"] == ["final.mp4", "trip-final.mp4"] and got["deleted_count"] == 2
    assert p.joinpath("highlights.json").is_file()  # a plan removes nothing


def test_delete_trashes_cutpoints_files_and_keeps_the_rest(
    ui_client: TestClient, bin_dir: Path
) -> None:
    p = make()
    write(p / "notes.txt", "mine")
    write(p / "clips" / "hand-made.mp4")
    write(p / "staged" / "copy.mp4")
    res = ui_client.request("DELETE", "/api/projects/trip", json={})
    assert res.status_code == 200, res.text
    assert res.json() == {"deleted": "trip", "kept": 3}
    left = sorted(f.relative_to(p).as_posix() for f in p.rglob("*") if f.is_file())
    assert left == ["clips/hand-made.mp4", "notes.txt", "staged/copy.mp4"]
    assert any("final" in f.name for f in bin_dir.iterdir())
    assert ui_client.get("/api/projects/trip").status_code == 404  # no longer a project


def test_a_project_with_only_cutpoint_files_goes_whole(
    ui_client: TestClient, bin_dir: Path
) -> None:
    p = make()
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert not p.exists()  # the folder goes once it is empty, the run lock file included
    assert {f.name.split("-", 1)[1] for f in bin_dir.iterdir()} >= {"ui", "clips", "final.mp4"}
    assert ui_client.get("/api/projects").json()["projects"] == []
    assert not ui.DELETING


def test_permanent_deletes_without_the_trash(ui_client: TestClient, bin_dir: Path) -> None:
    p = make()
    write(p / "notes.txt", "mine")
    res = ui_client.request("DELETE", "/api/projects/trip", json={"permanent": True})
    assert res.status_code == 200, res.text
    assert not list(bin_dir.iterdir())
    assert [f.name for f in p.iterdir()] == ["notes.txt"]


def test_a_project_made_again_under_the_same_name_keeps_nothing_of_the_deleted_one(
    ui_client: TestClient,
) -> None:
    make()
    make("other")
    app = ui_client.app
    old, other = ui.project_dir("trip"), ui.project_dir("other")
    previews, planners = app.state.previews, app.state.planners
    for q in (old, other):
        previews.outcomes[(q, "h001")] = {"state": "failed", "reason": "x", "ident": "h001-a"}
        previews.done[(q, "h001")] = {"ident": "h001-a", "name": "h001-a"}
        previews.memo[q] = ("sig", {})
        previews.ready_memo[q] = (0.0, "sig", {})
        previews.pruned.add(q)
        planners[q] = object()
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    make()
    p = ui.project_dir("trip")
    assert (p, "h001") not in previews.outcomes and (p, "h001") not in previews.done
    assert p not in previews.memo and p not in previews.ready_memo
    assert p not in previews.pruned and p not in planners
    # another project keeps what this run knows of it
    assert (other, "h001") in previews.outcomes and (other, "h001") in previews.done
    assert other in previews.memo and other in previews.ready_memo
    assert other in previews.pruned and other in planners


def test_permanent_must_be_a_boolean(ui_client: TestClient) -> None:
    make()
    res = ui_client.request("DELETE", "/api/projects/trip", json={"permanent": "yes"})
    assert res.status_code == 400


def test_no_trash_answers_with_a_code_and_deletes_nothing_more(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = make()

    def refuse(paths: list[str]) -> None:
        raise projectdelete.send2trash.TrashPermissionError(paths[0])

    monkeypatch.setattr(projectdelete.send2trash, "send2trash", refuse)
    res = ui_client.request("DELETE", "/api/projects/trip", json={})
    assert res.status_code == 409
    assert res.json()["code"] == "no_trash"
    assert "Trash" in res.json()["detail"]
    assert not ui.DELETING
    assert (p / "highlights.json").is_file()
    res = ui_client.request("DELETE", "/api/projects/trip", json={"permanent": True})
    assert res.status_code == 200, res.text
    assert not p.exists()


def test_a_foreign_manifest_keeps_its_folder(ui_client: TestClient) -> None:
    p = make()
    write(p / "mine" / "manifest.json", json.dumps({"generator": "other", "clips": []}))
    write(p / "mine" / "a.mp4")
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert (p / "mine" / "a.mp4").is_file() and (p / "mine" / "manifest.json").is_file()


def test_a_manifest_cannot_name_a_file_outside_its_folder(
    ui_client: TestClient, tmp_path: Path
) -> None:
    p = make()
    outside = write(tmp_path / "outside.mp4")
    manifest(p / "clips", ["../../outside.mp4", "001_a.mp4"])
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert outside.is_file()


@pytest.mark.parametrize("what", ["run", "clips", "job", "progress"])
def test_a_running_job_or_lock_refuses_with_409(
    ui_client: TestClient, what: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = make()
    if what == "run":
        ctx = folders.swap_lock(p / runstate.LOCK_NAME)
    elif what == "clips":
        ctx = folders.swap_lock(p / "clips")
    elif what == "job":
        job = ui.Job("trip", "clips")  # running until it ends
        ui_client.app.state.jobs.jobs[job.id] = job
        ctx = None
    else:
        ctx = None
        write(
            p / "progress.json",
            json.dumps({"command": "analyze", "status": "running", "pid": 1, "host": "elsewhere"}),
        )
    if ctx is not None:
        with ctx:
            res = ui_client.request("DELETE", "/api/projects/trip")
    else:
        res = ui_client.request("DELETE", "/api/projects/trip")
    assert res.status_code == 409, res.text
    assert (p / "highlights.json").is_file() and (p / "clips" / "001_a.mp4").is_file()


def test_a_job_of_another_project_does_not_refuse(ui_client: TestClient) -> None:
    make()
    job = ui.Job("other", "clips")
    ui_client.app.state.jobs.jobs[job.id] = job
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200


def test_a_link_out_of_the_project_is_not_followed(ui_client: TestClient, tmp_path: Path) -> None:
    p = make()
    elsewhere = tmp_path / "elsewhere"
    write(elsewhere / "precious.mp4")
    write(elsewhere / "highlights.json")
    link_dir(p / "ui" / "linked", elsewhere)  # inside a folder Cutpoint owns
    link_dir(p / "pano_paths", elsewhere)  # a Cutpoint folder that is itself a link
    link_file(p / "events.json", elsewhere / "highlights.json")
    plan = ui_client.get("/api/projects/trip/delete-plan").json()
    assert {"ui/linked/", "pano_paths/", "events.json"} <= set(plan["kept"])
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert (elsewhere / "precious.mp4").is_file() and (elsewhere / "highlights.json").is_file()
    assert (p / "ui" / "linked").exists() and not (p / "ui" / "meta.json").exists()
    assert not (p / "highlights.json").exists() and not (p / "clips").exists()


def test_a_linked_clips_folder_is_not_followed(ui_client: TestClient, tmp_path: Path) -> None:
    p = make()
    elsewhere = tmp_path / "elsewhere"
    write(elsewhere / "001_a.mp4")
    manifest(elsewhere, ["001_a.mp4"])
    link_dir(p / "more_clips", elsewhere)
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert (elsewhere / "001_a.mp4").is_file() and (elsewhere / "manifest.json").is_file()


def test_unknown_project_is_404(ui_client: TestClient) -> None:
    assert ui_client.request("DELETE", "/api/projects/nope").status_code == 404
    assert ui_client.get("/api/projects/nope/delete-plan").status_code == 404


def test_delete_needs_the_session_token(ui_client: TestClient) -> None:
    make()
    res = ui_client.request("DELETE", "/api/projects/trip", headers={"X-Cutpoint-Token": "wrong"})
    assert res.status_code == 403


def test_a_plain_permission_error_is_not_a_missing_trash(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = make()

    def locked(paths: list[str]) -> None:
        raise PermissionError(13, "Access is denied", paths[0])

    monkeypatch.setattr(projectdelete.send2trash, "send2trash", locked)
    res = ui_client.request("DELETE", "/api/projects/trip")
    assert res.status_code == 409
    assert "code" not in res.json() and "in use" in res.json()["detail"]
    assert (p / "highlights.json").is_file() and (p / "clips" / "001_a.mp4").is_file()
    assert not ui.DELETING


@pytest.fixture
def windows_probe(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The Windows probes of :func:`projectdelete.windows_blocker`, answering for a healthy
    fixed drive C: with a 1 GB bin (128 GB volume); a test changes one answer."""
    probe = SimpleNamespace(
        drive=3,
        dos_target=r"\Device\HarddiskVolume3",
        nuke=0,
        mb=1024,
        guid="{1}",
        total_mb=131072,
        used=0,
        policy=(False, None),
        settings_missing=False,
    )
    monkeypatch.setattr(projectdelete, "_mount_root", lambda path: "C:\\")
    monkeypatch.setattr(projectdelete, "_drive_type", lambda root: probe.drive)
    monkeypatch.setattr(projectdelete, "_dos_target", lambda drive: probe.dos_target)
    monkeypatch.setattr(projectdelete, "_volume_total_mb", lambda root: probe.total_mb)
    monkeypatch.setattr(projectdelete, "_bin_used", lambda root: probe.used)
    monkeypatch.setattr(projectdelete, "_policy", lambda: probe.policy)

    def guid(root: str) -> str:
        if probe.guid is None:
            raise OSError("no volume")
        return probe.guid

    def settings(g: str) -> tuple[int, int]:
        if probe.settings_missing:
            raise FileNotFoundError("no bin settings")
        if probe.nuke is None:
            raise PermissionError("denied")
        return probe.nuke, probe.mb

    monkeypatch.setattr(projectdelete, "_volume_guid", guid)
    monkeypatch.setattr(projectdelete, "_bin_settings", settings)
    return probe


MB100 = 100 * 1024**2
ITEM = Path("C:/p/ui")


def test_windows_probe_trusts_a_fixed_drive_with_room(windows_probe: SimpleNamespace) -> None:
    assert projectdelete.windows_blocker([(ITEM, 1024)]) is None
    assert projectdelete.windows_blocker([(ITEM, 900 * 1024**2)]) is None  # under 90% of 1 GB


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"drive": 4}, "not a fixed drive"),  # a network drive (removable: the same branch)
        ({"nuke": 1}, "delete files at once"),
        ({"mb": 10}, "bigger than the room"),
        ({"nuke": None}, "cannot tell"),
        ({"guid": None}, "cannot tell"),
        ({"policy": (True, None)}, "Group Policy"),
        ({"used": 1000 * 1024**2}, "bigger than the room"),
        ({"policy": (False, 1), "total_mb": 500}, "bigger than the room"),
    ],
)
def test_windows_probe_refuses_what_the_bin_may_delete_for_good(
    windows_probe: SimpleNamespace, change: dict, text: str
) -> None:
    for k, v in change.items():
        setattr(windows_probe, k, v)
    assert text in projectdelete.windows_blocker([(ITEM, MB100)])


def test_two_items_that_fit_alone_but_not_together_are_refused(
    windows_probe: SimpleNamespace,
) -> None:
    half = 600 * 1024**2  # each under 90% of 1 GB, together over it
    assert projectdelete.windows_blocker([(ITEM, half)]) is None
    assert "bigger than the room" in projectdelete.windows_blocker([(ITEM, half), (ITEM, half)])


def test_a_volume_without_settings_uses_the_windows_default(
    windows_probe: SimpleNamespace,
) -> None:
    windows_probe.settings_missing = True  # 128 GB volume: default bin = 4096 + 90112 * 5 // 100 MB
    assert projectdelete.windows_blocker([(ITEM, 1024**3)]) is None
    assert "bigger than the room" in projectdelete.windows_blocker([(ITEM, 9 * 1024**3)])
    windows_probe.total_mb = 1000  # small volume: 10% = 100 MB, 90 MB usable
    assert projectdelete.windows_blocker([(ITEM, 80 * 1024**2)]) is None
    assert "bigger than the room" in projectdelete.windows_blocker([(ITEM, 95 * 1024**2)])
    assert projectdelete._default_capacity_mb(40960) == 4096
    assert projectdelete._default_capacity_mb(131072) == 4096 + 90112 * 5 // 100


@pytest.mark.skipif(sys.platform != "win32", reason="Windows UNC paths")
def test_windows_probe_refuses_a_network_share(windows_probe: SimpleNamespace) -> None:
    assert "network share" in projectdelete.windows_blocker([(Path("//nas/share/p/ui"), 1)])


@pytest.mark.parametrize(
    ("target", "blocked"),
    [
        (r"\??\H:\scratch\p", True),
        (r"\Device\HarddiskVolume3", False),
        ("\\Device\\LanmanRedirector\\;Z:000\nas\\share", False),
        ("", False),  # QueryDosDevice failed: behaves as before
    ],
)
def test_windows_probe_refuses_a_subst_drive(
    windows_probe: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, target: str, blocked: bool
) -> None:
    # abspath of a drive path is the same on every OS only on Windows, so make it explicit
    monkeypatch.setattr(projectdelete.os.path, "splitdrive", lambda p: ("X:", p[2:]))
    windows_probe.dos_target = target
    reason = projectdelete.windows_blocker([(Path("X:/p/ui"), 1)])
    if blocked:
        assert "substituted drive (subst)" in reason
    else:
        assert reason is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows subst drives")
def test_a_real_subst_drive_is_refused(tmp_path: Path) -> None:
    import ctypes  # noqa: PLC0415
    import string  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    used = ctypes.windll.kernel32.GetLogicalDrives()  # type: ignore[attr-defined]
    free = [c for i, c in enumerate(string.ascii_uppercase) if not used >> i & 1][::-1]
    if not free:
        pytest.skip("no free drive letter")
    drive = f"{free[0]}:"
    (tmp_path / "p").mkdir()
    try:
        subprocess.run(["subst", drive, str(tmp_path)], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip(f"subst {drive} failed")
    try:
        reason = projectdelete.windows_blocker([(Path(drive + r"\p"), 1)])
        assert reason is not None and "subst" in reason
        assert projectdelete.windows_blocker([(tmp_path / "p", 1)]) != reason
    finally:
        subprocess.run(["subst", drive, "/D"], check=False)


def test_two_items_over_the_limit_together_ask_first_and_move_nothing(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch, windows_probe: SimpleNamespace
) -> None:
    p = make()
    write(p / "ui" / "proxy" / "big.mp4", b"x" * 700)
    moved: list[str] = []
    monkeypatch.setattr(projectdelete.send2trash, "send2trash", moved.extend)
    monkeypatch.setattr(projectdelete, "trash_blocker", projectdelete.windows_blocker)
    windows_probe.mb = 1  # 1 MB bin: 90% of it is about 920 KB
    plan = projectdelete.make_plan(p)
    plan.sizes = [600 * 1024] * len(plan.items)  # each alone fits, the sum does not
    with pytest.raises(projectdelete.NoTrash):
        projectdelete.remove_items(plan, permanent=False)
    assert moved == [] and (p / "highlights.json").is_file()


def test_macos_volume_without_a_trash_is_a_missing_trash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = write(tmp_path / "x.txt")
    plan = projectdelete.Plan(items=[f], sizes=[1])

    def refuse(paths: list[str]) -> None:
        raise OSError("The item can't be moved to the trash on this volume")

    monkeypatch.setattr(projectdelete.send2trash, "send2trash", refuse)
    monkeypatch.setattr(projectdelete, "trash_blocker", lambda items: None)
    monkeypatch.setattr(projectdelete.sys, "platform", "darwin")
    with pytest.raises(projectdelete.NoTrash):
        projectdelete.remove_items(plan, permanent=False)
    monkeypatch.setattr(projectdelete.sys, "platform", "linux")  # other systems: not a NoTrash
    with pytest.raises(OSError) as err:
        projectdelete.remove_items(plan, permanent=False)
    assert not isinstance(err.value, projectdelete.NoTrash) and f.is_file()


def test_a_list_filename_names_the_file_in_use(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = make()

    def locked(paths: list[str]) -> None:
        raise PermissionError(13, "Access is denied", paths)  # a list, as a batch call gives

    monkeypatch.setattr(projectdelete.send2trash, "send2trash", locked)
    res = ui_client.request("DELETE", "/api/projects/trip")
    assert res.status_code == 409 and "in use" in res.json()["detail"]
    assert (p / "highlights.json").is_file()


@pytest.mark.parametrize("reason", ["on a network share", "a drive is not fixed", "too big"])
def test_a_blocked_trash_asks_first_and_removes_nothing(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    p = make()
    moved: list[str] = []
    monkeypatch.setattr(projectdelete.send2trash, "send2trash", moved.extend)
    monkeypatch.setattr(projectdelete, "trash_blocker", lambda items: reason)
    res = ui_client.request("DELETE", "/api/projects/trip", json={})
    assert res.status_code == 409 and res.json()["code"] == "no_trash"
    assert reason in res.json()["detail"]
    assert moved == [] and (p / "highlights.json").is_file() and (p / "ui" / "meta.json").is_file()
    # the second, explicit confirm deletes for good without the probe
    res = ui_client.request("DELETE", "/api/projects/trip", json={"permanent": True})
    assert res.status_code == 200 and not p.exists()


def test_a_container_has_no_trash(ui_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    p = make()
    monkeypatch.setattr(projectdelete, "trash_blocker", REAL_TRASH_BLOCKER)
    monkeypatch.setattr(projectdelete, "in_container", lambda: True)
    res = ui_client.request("DELETE", "/api/projects/trip", json={})
    assert res.status_code == 409 and res.json()["code"] == "no_trash"
    assert "container" in res.json()["detail"] and (p / "highlights.json").is_file()


def test_a_source_video_of_any_project_is_never_deleted(ui_client: TestClient) -> None:
    p = make()
    other = ui.projects_dir() / "other"
    meta = {"name": "Other", "source": str(p / "ui" / "proxy")}
    write(other / "ui" / "meta.json", json.dumps(meta))
    plan = ui_client.get("/api/projects/trip/delete-plan").json()
    assert "ui/ (used as a source by project Other)" in plan["kept"]
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert (p / "ui" / "proxy" / "a.mp4").is_file()
    assert not (p / "highlights.json").exists() and not (p / "clips").exists()


def test_a_source_listed_by_an_analysis_stays_too(ui_client: TestClient) -> None:
    p = make()
    other = ui.projects_dir() / "cli"
    hl = {"sources": [{"path": str(p / "final.mp4")}], "settings": {}}
    write(other / "highlights.json", json.dumps(hl))
    plan = ui_client.get("/api/projects/trip/delete-plan").json()
    assert "final.mp4 (used as a source by project cli)" in plan["kept"]
    assert plan["deleted"] == ["trip-final.mp4"]


def test_the_run_lock_is_held_while_files_go_and_the_project_is_marked(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = make()
    seen: dict[str, bool] = {}
    real = projectdelete.remove_items

    def spy(plan, *, permanent):
        seen["locked"] = folders.lock_held(p / runstate.LOCK_NAME)
        seen["marked"] = p in ui.DELETING
        with pytest.raises(HTTPException) as err:
            ui.project_dir("trip")
        seen["409"] = err.value.status_code == 409
        real(plan, permanent=permanent)

    monkeypatch.setattr(projectdelete, "remove_items", spy)
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert seen == {"locked": True, "marked": True, "409": True}
    assert not ui.DELETING and not p.exists()


def test_the_jobs_lock_is_free_while_files_go_and_no_job_starts(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Moving GBs to the trash must not hold the jobs lock (#1085, A272); a job request that
    # resolved the folder just before the delete marked it is still refused.
    footage = tmp_path / "footage"
    write(footage / "a.mp4", "x")
    res = ui_client.post("/api/projects", json={"name": "Trip", "source": str(footage)})
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    moving, release = threading.Event(), threading.Event()
    real = projectdelete.remove_items

    def slow(plan, *, permanent):
        moving.set()
        release.wait(10)
        real(plan, permanent=permanent)

    spawned: list[list[str]] = []

    def no_spawn(job, args):
        spawned.append(args)
        raise OSError("not in tests")

    monkeypatch.setattr(projectdelete, "remove_items", slow)
    monkeypatch.setattr(ui.Jobs, "_popen", staticmethod(no_spawn))
    # as resolved before the delete began: the 409 of project_dir is not seen
    monkeypatch.setattr(jobs_routes, "project_dir", lambda p: ui.projects_dir() / p)
    out: dict[str, int] = {}
    t = threading.Thread(
        target=lambda: out.update(
            code=ui_client.request("DELETE", f"/api/projects/{pid}").status_code
        )
    )
    t.start()
    try:
        assert moving.wait(10)
        lock = ui_client.app.state.jobs.lock
        free = lock.acquire(timeout=1)
        if free:
            lock.release()
        assert free, "the jobs lock is held while the files are moved"
        res = ui_client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
        assert res.status_code == 409, res.text
        assert not spawned
    finally:
        release.set()
        t.join(20)
    assert out == {"code": 200}
    assert not ui.DELETING


def test_a_job_for_a_folder_deleted_after_it_was_resolved_is_refused(
    ui_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A job request that resolved the folder before a delete and reaches the jobs lock after
    # the delete ended (the mark is gone) must not start on the vanished folder.
    footage = tmp_path / "footage"
    write(footage / "a.mp4", "x")
    res = ui_client.post("/api/projects", json={"name": "Trip", "source": str(footage)})
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    spawned: list[list[str]] = []

    def no_spawn(job, args):
        spawned.append(args)
        raise OSError("not in tests")

    def deleted_meanwhile() -> None:
        assert ui_client.request("DELETE", f"/api/projects/{pid}").status_code == 200

    monkeypatch.setattr(ui.Jobs, "_popen", staticmethod(no_spawn))
    real_start = ui.Jobs.start

    def start_after_delete(self, *args, **kwargs):
        deleted_meanwhile()  # between the route's resolve and the check under the lock
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(ui.Jobs, "start", start_after_delete)
    res = ui_client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 404, res.text
    assert not spawned
    assert not (ui.projects_dir() / pid).exists()


def test_qa_folders_go_and_hud_presets_stay_with_a_reason(ui_client: TestClient) -> None:
    p = make()
    write(p / "clips_qa" / "qa.json", "{}")
    write(p / "clips_qa" / "flagged_01.jpg")
    write(p / "clips_qa" / "mine.txt")
    write(p / "hud_presets" / "game.json", "{}")
    plan = ui_client.get("/api/projects/trip/delete-plan").json()
    assert sorted(plan["kept"]) == [
        "clips_qa/mine.txt",
        "hud_presets/ (HUD presets you may have edited)",
    ]
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    assert sorted(f.name for f in p.rglob("*") if f.is_file()) == ["game.json", "mine.txt"]


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions")
def test_junctions_are_listed_and_never_followed(ui_client: TestClient, tmp_path: Path) -> None:
    import _winapi  # noqa: PLC0415

    p = make()
    elsewhere = tmp_path / "elsewhere"
    write(elsewhere / "precious.mp4")
    write(elsewhere / "001_a.mp4")
    manifest(elsewhere, ["001_a.mp4"])
    _winapi.CreateJunction(str(elsewhere), str(p / "ui" / "j"))
    _winapi.CreateJunction(str(elsewhere), str(p / "more_clips"))
    plan = ui_client.get("/api/projects/trip/delete-plan").json()
    assert {"ui/j/", "more_clips/"} <= set(plan["kept"])
    assert ui_client.request("DELETE", "/api/projects/trip").status_code == 200
    survivors = sorted(f.name for f in elsewhere.iterdir())
    assert survivors == ["001_a.mp4", "manifest.json", "precious.mp4"]
    assert (p / "ui" / "j").exists() and (p / "more_clips").exists()
    assert not (p / "highlights.json").exists()


# Placeholders of the AGENTS.md "Project files" table, filled in as a run would
PLACEHOLDERS = {"<label>": "x", "<name>": "trip", "<id>": "run1", "<folder>": "clips"}


def agents_project_files() -> list[str]:
    """The names in the first column of AGENTS.md's "Project files" table."""
    text = (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text(encoding="utf-8")
    table = text.split("## Project files", 1)[1].split("\n## ", 1)[0]
    names: list[str] = []
    for row in table.splitlines():  # the heading and |---| rows have no backticks
        if row.startswith("|"):
            names += re.findall(r"`([^`]+)`", row.split("|")[1])
    return names


def test_every_project_file_agents_md_names_is_deleted_with_the_project() -> None:
    """A file the table lists but the delete does not know keeps the project folder (#1085)."""
    names = agents_project_files()
    assert {"highlights.json", "highlights.before-rerank.json"} <= set(names)  # all rows read
    assert len(names) > 15
    p = ui.projects_dir() / "trip"
    p.mkdir(parents=True)
    for name in names:
        if re.fullmatch(r"\.\w+", name):  # "(or `.fcpxml` / `.otio`)": a suffix, not a file
            continue
        for k, v in PLACEHOLDERS.items():
            name = name.replace(k, v)
        if name.endswith("/"):
            write(p / name / "a.mp4")
        else:
            write(p / name, "{}")
    clips = sorted(f.name for f in (p / "clips").iterdir() if f.suffix != ".json")
    write(p / "clips" / "001_x.mp4")
    manifest(p / "clips", ["001_x.mp4"])
    assert clips == ["001_x.srt"]
    plan = projectdelete.make_plan(p)
    assert plan.kept == []


def test_the_rerank_backup_is_one_of_the_project_files() -> None:
    assert projectfiles.RERANK_BACKUP in projectdelete.OWN_FILES
