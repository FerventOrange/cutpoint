"""Media routes keep a project's folder check and source list briefly (#1014): a browser opens a
fragmented MP4 with hundreds of Range requests, and on a network share each folder check cost
~10 ms. Also the page's hints for a slow start: the project folder on a network drive, a
fragmented MP4."""

from __future__ import annotations

import shutil
import struct
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from conftest import media_info
from cutpoint import projectdelete, stage, ui

PORT = 8765
DATA = b"0123456789" * 100


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    # never trash a real file: deletes move into a stand-in bin; never probe the host's trash
    monkeypatch.setattr(
        projectdelete.send2trash,
        "send2trash",
        lambda paths: [shutil.move(p, bin_ / f"{i}-{Path(p).name}") for i, p in enumerate(paths)],
    )
    monkeypatch.setattr(projectdelete, "trash_blocker", lambda items: None)
    monkeypatch.setattr(ui, "probe", lambda path: media_info(Path(path), duration=10.0))
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Counts of project folder checks and source listings."""
    calls = {"project_dir": 0, "entries": 0}
    real_dir, real_entries = ui.project_dir, ui.Sources.entries

    def project_dir(pid: str) -> Path:
        calls["project_dir"] += 1
        return real_dir(pid)

    def entries(self: ui.Sources, p: Path) -> list[dict]:
        calls["entries"] += 1
        return real_entries(self, p)

    monkeypatch.setattr(ui, "project_dir", project_dir)
    monkeypatch.setattr(ui.Sources, "entries", entries)
    return calls


def make_project(client: TestClient, tmp_path: Path, data: bytes = DATA) -> tuple[str, str, Path]:
    """``(project id, source id, source file)`` of a project with one MP4 that plays as is."""
    folder = tmp_path / "footage"
    folder.mkdir()
    source = folder / "a.mp4"
    source.write_bytes(data)
    res = client.post("/api/projects", json={"name": "My Trip", "source": str(folder)})
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    return pid, sid, source


def media_url(client: TestClient, pid: str, sid: str) -> str:
    return f"/api/projects/{pid}/sources/{sid}?t={client.headers['X-Cutpoint-Token']}"


def box(kind: bytes, body: bytes = b"") -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def test_range_requests_within_the_ttl_check_the_folder_once(
    client: TestClient, tmp_path: Path, counted: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _ = make_project(client, tmp_path)
    monkeypatch.setattr(ui, "MEDIA_LOOKUP_S", 3600.0)  # within the TTL however slow the runner
    url = media_url(client, pid, sid)
    counted.update(project_dir=0, entries=0)
    for i in range(50):
        res = client.get(url, headers={"Range": f"bytes={i}-{i + 9}"})
        assert (res.status_code, res.content) == (206, DATA[i : i + 10])
    assert counted == {"project_dir": 1, "entries": 1}


def test_after_the_ttl_the_folder_is_checked_again(
    client: TestClient, tmp_path: Path, counted: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _ = make_project(client, tmp_path)
    monkeypatch.setattr(ui, "MEDIA_LOOKUP_S", 0.0)
    counted.update(project_dir=0, entries=0)
    for _ in range(3):
        assert client.get(media_url(client, pid, sid)).status_code == 200
    assert counted["project_dir"] == 3


def test_per_request_cost_drops_with_a_slow_folder_check(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured share: ~13 ms per folder check and listing. Kept, N Range requests pay it once
    per ``MEDIA_LOOKUP_S`` instead of N times (600 requests: ~16 s before)."""
    pid, sid, _ = make_project(client, tmp_path)
    url = media_url(client, pid, sid)
    real_dir = ui.project_dir
    slept: list[float] = []

    def slow(pid: str) -> Path:
        time.sleep(0.013)
        slept.append(0.013)
        return real_dir(pid)

    monkeypatch.setattr(ui, "project_dir", slow)
    monkeypatch.setattr(ui, "MEDIA_LOOKUP_S", 3600.0)  # a slow runner takes over 2 s for n
    n = 200
    for i in range(n):
        assert client.get(url, headers={"Range": f"bytes={i}-{i}"}).status_code == 206
    assert len(slept) == 1  # n x 13 ms = 2.6 s of folder checks before; 13 ms now
    assert sum(slept) < n * 0.013 / 100


def test_an_unknown_project_is_404_and_a_new_one_is_found_at_once(
    client: TestClient, tmp_path: Path
) -> None:
    token = client.headers["X-Cutpoint-Token"]
    assert client.get(f"/api/projects/nope/sources/abc?t={token}").status_code == 404
    pid, sid, _ = make_project(client, tmp_path)
    assert client.get(f"/api/projects/{pid}/sources/abc?t={token}").status_code == 404
    assert client.get(media_url(client, pid, sid)).status_code == 200


def test_a_project_being_deleted_is_409_even_while_kept(client: TestClient, tmp_path: Path) -> None:
    pid, sid, _ = make_project(client, tmp_path)
    url = media_url(client, pid, sid)
    assert client.get(url).status_code == 200  # kept now
    p = ui.projects_dir() / pid
    ui.DELETING.add(p)
    try:
        assert client.get(url).status_code == 409
        window = (
            f"/api/projects/{pid}/sources/{sid}/window/0-100?t={client.headers['X-Cutpoint-Token']}"
        )
        assert client.get(window).status_code == 409
    finally:
        ui.DELETING.discard(p)


def test_a_project_deleted_in_the_app_is_404_at_once(client: TestClient, tmp_path: Path) -> None:
    pid, sid, _ = make_project(client, tmp_path)
    url = media_url(client, pid, sid)
    assert client.get(url).status_code == 200  # kept now
    res = client.request("DELETE", f"/api/projects/{pid}", json={"permanent": True})
    assert res.status_code == 200, res.text
    assert client.get(url).status_code == 404


def test_a_project_deleted_outside_the_app_is_404_after_the_ttl(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guarantee: an outside delete shows within ``MEDIA_LOOKUP_S`` (2 s)."""
    pid, sid, _ = make_project(client, tmp_path)
    url = media_url(client, pid, sid)
    assert client.get(url).status_code == 200
    shutil.rmtree(ui.projects_dir() / pid)
    monkeypatch.setattr(ui, "MEDIA_LOOKUP_S", 0.0)  # the TTL passed
    assert client.get(url).status_code == 404


def test_the_kept_answer_is_per_projects_folder(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _ = make_project(client, tmp_path)
    url = media_url(client, pid, sid)
    assert client.get(url).status_code == 200
    other = tmp_path / "other-projects"
    other.mkdir()
    monkeypatch.setattr(ui, "projects_dir", lambda: other)
    assert client.get(url).status_code == 404


# ---------------------------------------------------------------- slow-start hints


@pytest.mark.parametrize(
    ("data", "expect"),
    [
        (box(b"ftyp", b"isom") + box(b"moov", box(b"mvex")) + box(b"moof") + box(b"mdat"), True),
        (box(b"ftyp") + box(b"moov") + box(b"sidx") + box(b"moof") + box(b"mdat"), False),
        (box(b"ftyp") + box(b"moov") + box(b"mdat", b"x" * 100), False),  # flat, faststart
        (box(b"ftyp") + box(b"mdat", b"x" * 100) + box(b"moov"), False),  # flat, moov at the end
        (
            box(b"ftyp")
            + struct.pack(">I", 1)
            + b"free"
            + struct.pack(">Q", 24)
            + b"x" * 8
            + box(b"moof"),
            True,
        ),  # a 64-bit box size
        (box(b"ftyp") + struct.pack(">I", 0) + b"mdat" + b"x" * 20, False),  # to the end
        (struct.pack(">I", 3) + b"ftyp" + box(b"moof"), False),  # a broken size
        (b"\x1a\x45\xdf\xa3 webm", False),
        (b"", False),
    ],
)
def test_fragmented_mp4(tmp_path: Path, data: bytes, expect: bool) -> None:
    f = tmp_path / "v.mp4"
    f.write_bytes(data)
    assert ui.fragmented_mp4(f) is expect


def test_fragmented_mp4_of_a_missing_file_is_false(tmp_path: Path) -> None:
    assert ui.fragmented_mp4(tmp_path / "gone.mp4") is False


def no_drive_type(root: str) -> int:
    raise OSError("no drive type")  # Windows then falls back to stage.is_remote


def test_the_project_lists_its_slow_start_hints(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ui, "_REMOTE", {})
    monkeypatch.setattr(projectdelete, "_drive_type", no_drive_type)  # tested apart
    monkeypatch.setattr(stage, "is_remote", lambda path: True)
    fragmented = box(b"ftyp") + box(b"moov", box(b"mvex")) + box(b"moof") + box(b"mdat")
    pid, sid, _ = make_project(client, tmp_path, fragmented)
    project = client.get(f"/api/projects/{pid}").json()
    assert project["network"] is True
    assert project["sources"][0]["fragmented"] is None  # not probed yet: the listing never reads
    assert client.get(media_url(client, pid, sid)).status_code == 200  # probes it
    assert client.get(f"/api/projects/{pid}").json()["sources"][0]["fragmented"] is True


def test_a_flat_source_on_local_storage_has_no_hints(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ui, "_REMOTE", {})
    monkeypatch.setattr(projectdelete, "_drive_type", no_drive_type)  # tested apart
    monkeypatch.setattr(stage, "is_remote", lambda path: False)
    pid, sid, _ = make_project(client, tmp_path, box(b"ftyp") + box(b"moov") + box(b"mdat"))
    assert client.get(media_url(client, pid, sid)).status_code == 200
    project = client.get(f"/api/projects/{pid}").json()
    assert (project["network"], project["sources"][0]["fragmented"]) == (False, False)


def test_a_check_that_raced_an_in_app_delete_is_not_kept(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A media request that passed the folder check while the delete finished keeps nothing, so
    the next one is 404 rather than a kept 200."""
    pid, sid, _ = make_project(client, tmp_path)
    url = media_url(client, pid, sid)
    lookups: ui.MediaLookups = client.app.state.lookups
    real_dir = ui.project_dir

    def racing(pid: str) -> Path:
        p = real_dir(pid)
        lookups.forget(p)  # the delete ends between the check and the store
        return p

    monkeypatch.setattr(ui, "project_dir", racing)
    assert client.get(url).status_code == 200  # served from the check it made
    assert lookups.kept == {}
    monkeypatch.setattr(ui, "project_dir", real_dir)
    shutil.rmtree(ui.projects_dir() / pid)
    assert client.get(url).status_code == 404


@pytest.mark.parametrize(
    ("kind", "expect"), [(3, False), (2, True), (4, True), (6, False), (0, False)]
)
def test_on_windows_the_drive_type_says_network(
    monkeypatch: pytest.MonkeyPatch, kind: int, expect: bool
) -> None:
    """Fixed (3) or RAM (6) drives are local, also F: and beyond; removable (2) and remote (4,
    also a UNC path) are named; an unknown type (0) is not."""
    monkeypatch.setattr(ui, "_REMOTE", {})
    monkeypatch.setattr(ui.sys, "platform", "win32")
    roots: list[str] = []
    monkeypatch.setattr(ui.projectdelete, "_drive_type", lambda root: roots.append(root) or kind)
    monkeypatch.setattr(stage, "is_remote", lambda path: pytest.fail("not asked on Windows"))
    p = Path("F:/Projects/trip")
    assert ui.on_network(p) is expect
    assert roots == [p.anchor]


def test_on_windows_a_failed_drive_type_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "_REMOTE", {})
    monkeypatch.setattr(ui.sys, "platform", "win32")

    def broken(root: str) -> int:
        raise OSError("no kernel32")

    monkeypatch.setattr(ui.projectdelete, "_drive_type", broken)
    monkeypatch.setattr(stage, "is_remote", lambda path: True)
    assert ui.on_network(Path("F:/Projects/trip")) is True
