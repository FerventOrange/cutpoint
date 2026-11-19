"""Projects in ``cutpoint ui``: creating them, 360 projects, stitched exports, director previews."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import patch_ui, set_config, write
from cutpoint import cache, capabilities, cli, errors, ffmpeg_utils, folders, procs, ui
from cutpoint.assistant import cliproxy
from cutpoint.pano import candidates as pano_candidates
from cutpoint.pano import director as pano_director
from cutpoint.pano import project as pano_project
from cutpoint.pano import render as pano_render
from cutpoint.pano import stitch as pano_stitch
from cutpoint.ui.routes import assistant as assistant_routes
from cutpoint.utils import find_videos
from pano_helpers import x3clip
from test_utils import MPEG_EXTS
from ui_helpers import (
    PORT,
    X3_00,
    X3_10,
    X3_CLIPS,
    X3_LRV,
    FakeProc,
    JobSpy,
    cli_project,
    make_project,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


@pytest.mark.parametrize(
    ("files", "folders", "pick", "status", "expect"),
    [
        (
            {"a.insv": b"", "notes.txt": b""},
            ["sub"],
            None,
            400,
            ["no supported videos found (found 1 .insv, 1 .txt)", "camera names"],
        ),
        ({X3_00: b"", X3_LRV: b"", "a.mp4": b""}, [], None, 400, ["mixes 360 footage"]),
        ({X3_10: b"", X3_LRV: b""}, [], None, 400, ["needs both files", "_00_.insv"]),
    ],
    ids=[
        "unnamed-insv",
        "mixed",
        "lrv-without-00",
    ],
)
def test_create_project_recognises_360_sources_and_refuses_them_clearly(
    client: TestClient,
    tmp_path: Path,
    files: dict[str, bytes],
    folders: list[str],
    pick: str | None,
    status: int,
    expect: list[str],
) -> None:
    """Regression for #321: a folder of .insv files said "no videos", not what was wrong."""
    folder = tmp_path / "footage"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    for name in folders:
        (folder / name).mkdir()
    source = folder / pick if pick else folder
    res = client.post("/api/projects", json={"name": "Trip", "source": str(source)})
    assert res.status_code == status, res.text
    for part in expect:
        assert part in res.json()["detail"]
    if status != 200:
        assert client.get("/api/projects").json() == {"projects": []}


@pytest.mark.parametrize(
    ("pick", "style", "meta"),
    [
        (None, None, {"kind": "360", "style": "action-cam"}),
        (X3_00, None, {"kind": "360", "style": "action-cam", "clips": ["20260101_100000_001"]}),
        (None, "action-cam", {"kind": "360", "style": "action-cam"}),
    ],
    ids=["folder-flat-style-becomes-pano-default", "file-keeps-its-clip", "action-cam-style"],
)
def test_create_project_makes_a_360_project_from_x3_footage(
    client: TestClient, tmp_path: Path, pick: str | None, style: str | None, meta: dict
) -> None:
    """#320: X3 sources become 360 projects; ``clips`` only for one picked file."""
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    body = {"name": "Trip", "source": str(folder / pick if pick else folder)}
    if style:
        body["style"] = style
    res = client.post("/api/projects", json=body)
    assert res.status_code == 200, res.text
    saved = ui.read_meta(ui.project_dir(res.json()["id"]))
    assert {k: saved.get(k) for k in meta} == meta
    assert ("clips" in saved) == ("clips" in meta)


def test_ui_styles_lists_action_cam() -> None:
    """#633: the telemetry-driven style is offered for standard footage too."""
    assert "action-cam" in ui.ui_styles()


def test_create_project_accepts_an_action_cam_style_on_flat_footage(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    (folder / "b.mp4").write_bytes(b"x")
    for source in (folder / "a.mp4", folder):  # one file, one folder
        body = {"name": "T", "source": str(source), "style": "action-cam"}
        res = client.post("/api/projects", json=body)
        assert res.status_code == 200, res.text
        meta = ui.read_meta(ui.project_dir(res.json()["id"]))
        assert meta["style"] == "action-cam" and meta.get("kind") != "360"


def test_create_project_accepts_auto_and_defaults_to_it(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An omitted style is Auto; the analyze job runs --style auto and the vision flags follow
    Settings (#919): --no-vision when it is off, --vision when on, neither when unset."""
    footage = tmp_path / "footage"
    footage.mkdir()
    (footage / "a.mp4").write_bytes(b"x")
    captured = JobSpy(monkeypatch, check=False).args
    for extra in ({}, {"style": "auto"}):
        res = client.post("/api/projects", json={"name": "T", "source": str(footage), **extra})
        assert res.status_code == 200, res.text
        pid = res.json()["id"]
        assert ui.read_meta(ui.project_dir(pid))["style"] == "auto"
        assert client.get(f"/api/projects/{pid}").json()["style"] == "auto"
    for cfg, want in ((None, (False, False)), (True, (True, False)), (False, (False, True))):
        set_config(home, **({} if cfg is None else {"vision": cfg}))
        captured.clear()
        job = {"action": "analyze"}
        assert client.post(f"/api/projects/{pid}/jobs", json=job).status_code == 200
        args = captured[0]
        assert args[args.index("--style") + 1] == "auto"
        assert ("--vision" in args, "--no-vision" in args) == want


def test_create_project_still_refuses_an_unknown_style_next_to_auto(
    client: TestClient, tmp_path: Path
) -> None:
    footage = tmp_path / "a.mp4"
    footage.write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "T", "source": str(footage), "style": "nope"})
    assert res.status_code == 400 and "unknown style" in res.json()["detail"]


def test_create_project_accepts_an_action_cam_style_for_several_inputs(
    client: TestClient, tmp_path: Path
) -> None:
    files = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for f in files:
        f.write_bytes(b"x")
    body = {"name": "T", "sources": [str(f) for f in files], "style": "action-cam"}
    res = client.post("/api/projects", json=body)
    assert res.status_code == 200, res.text
    meta = ui.read_meta(ui.project_dir(res.json()["id"]))
    assert meta["style"] == "action-cam" and len(meta["inputs"]) == 2


def test_create_project_from_several_inputs_lists_and_analyzes_them_all(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#484: several files and folders are the project's analyze inputs, in the order given."""
    one, two = tmp_path / "one", tmp_path / "two"
    for folder in (one, two):
        folder.mkdir()
    for f in (one / "a.mp4", one / "other.mp4", two / "b.mp4", two / "c.mkv"):
        f.write_bytes(b"x")
    inputs = [str(one / "a.mp4"), str(two)]
    res = client.post("/api/projects", json={"name": "Two", "sources": inputs, "style": "funny"})
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    meta = ui.read_meta(ui.project_dir(pid))
    assert (meta["source"], meta["inputs"]) == (inputs[0], inputs)
    assert "recursive" not in meta and "exclude" not in meta
    view = client.get(f"/api/projects/{pid}").json()
    # other.mp4 sits next to a picked file: not an input, so never listed or served
    assert [s["name"] for s in view["sources"]] == ["a.mp4", "b.mp4", "c.mkv"]
    assert view["preset_problem"] is None  # edit takes several inputs too

    captured = JobSpy(monkeypatch, check=False).args
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    ns = cli.build_parser().parse_args(captured[0][3:])
    assert (ns.inputs, ns.exclude, ns.recursive) == (inputs, None, False)


def test_create_project_from_a_folder_passes_recursive_and_exclude_to_analyze(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#484: Include sub-folders and Skip files matching are analyze --recursive and --exclude;
    the sources listed before analysis are the ones analyze would find."""
    footage = tmp_path / "footage"
    (footage / "sub").mkdir(parents=True)
    (footage / "Captures").mkdir()
    for rel in ("keep.mp4", "DVR_skip.mp4", "sub/nested.mp4", "sub/DVR_x.mp4", "Captures/c.mp4"):
        (footage / rel).write_bytes(b"x")
    exclude = ["*DVR*", "*/Captures/*"]
    body = {"name": "F", "sources": [str(footage)], "recursive": True, "exclude": exclude}
    res = client.post("/api/projects", json=body)
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    meta = ui.read_meta(ui.project_dir(pid))
    assert (meta["source"], meta["recursive"], meta["exclude"]) == (str(footage), True, exclude)
    assert "inputs" not in meta  # one input: source alone, as before
    view = client.get(f"/api/projects/{pid}").json()
    assert {s["name"] for s in view["sources"]} == {"keep.mp4", "nested.mp4"}
    assert set(map(Path, (s["path"] for s in view["sources"]))) == set(
        find_videos(footage, exclude=exclude, recursive=True)
    )
    assert "--recursive" in view["preset_problem"]  # edit takes neither option

    captured = JobSpy(monkeypatch, check=False).args
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert [a for a in captured[0] if a.startswith("--exclude")] == [
        "--exclude=*DVR*",
        "--exclude=*/Captures/*",
    ]
    ns = cli.build_parser().parse_args(captured[0][3:])
    assert (ns.inputs, ns.exclude, ns.recursive) == ([str(footage)], exclude, True)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "edit", "kind": "montage"})
    assert res.status_code == 400 and "--exclude or --recursive" in res.json()["detail"]


def test_analyze_again_keeps_options_a_command_line_reanalysis_of_a_ui_project_added(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    footage = tmp_path / "footage"
    footage.mkdir()
    (footage / "a.mp4").write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "F", "sources": [str(footage)]})
    pid = res.json()["id"]
    settings = {"inputs": [str(footage)], "exclude": ["*DVR*"], "recursive": True}
    hl = {"style": "funny", "settings": settings, "sources": []}
    cache.save_json(ui.project_dir(pid) / "highlights.json", hl)
    captured = JobSpy(monkeypatch, check=False).args
    assert client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"}).status_code == 200
    ns = cli.build_parser().parse_args(captured[0][3:])
    assert (ns.exclude, ns.recursive) == (["*DVR*"], True)


def test_a_folder_with_videos_only_in_sub_folders_needs_recursive(
    client: TestClient, tmp_path: Path
) -> None:
    footage = tmp_path / "footage"
    (footage / "day1").mkdir(parents=True)
    (footage / "day1" / "a.mp4").write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "F", "sources": [str(footage)]})
    assert res.status_code == 400
    body = {"name": "F", "sources": [str(footage)], "recursive": True}
    res = client.post("/api/projects", json=body)
    assert res.status_code == 200, res.text
    names = [s["name"] for s in client.get(f"/api/projects/{res.json()['id']}").json()["sources"]]
    assert names == ["a.mp4"]


@pytest.mark.parametrize(
    ("body", "expect"),
    [
        ({"sources": ["relative/clip.mp4"]}, "absolute path"),
        ({"sources": ["{footage}/a.mp4\x00.txt"]}, "absolute path"),
        ({"sources": ["{footage}", "{footage}"]}, "once"),
        ({"sources": ["{footage}", "{pano}"]}, "360 footage"),
        ({"sources": ["{mixed}"], "exclude": ["*.mp4"]}, "mixes 360 footage"),
        ({"sources": ["{footage}"], "exclude": ["*.mp4"]}, "no videos found that the exclude"),
    ],
    ids=["relative", "nul", "twice", "360-with-flat", "mixed-folder", "all-excluded"],
)
def test_create_project_refuses_bad_inputs_and_options(
    client: TestClient, tmp_path: Path, body: dict, expect: str
) -> None:
    """#484: every input is an absolute path to a video or folder, each once; nothing is made
    when one is refused."""
    footage, pano, mixed = tmp_path / "footage", tmp_path / "pano", tmp_path / "mixed"
    for folder in (footage, pano, mixed):
        folder.mkdir()
    (footage / "a.mp4").write_bytes(b"x")
    for name in X3_CLIPS:
        (pano / name).write_bytes(b"")
        (mixed / name).write_bytes(b"")
    (mixed / "a.mp4").write_bytes(b"x")

    def fill(v: object) -> object:
        if isinstance(v, list):
            return [fill(x) for x in v]
        if isinstance(v, str) and v.startswith("{"):
            return str(Path(v.format(footage=footage, pano=pano, mixed=mixed)))
        return v

    res = client.post("/api/projects", json={"name": "T", **{k: fill(v) for k, v in body.items()}})
    assert res.status_code == 400, res.text
    assert expect in res.json()["detail"]
    assert client.get("/api/projects").json() == {"projects": []}


def test_create_project_reads_sources_as_pasted(client: TestClient, tmp_path: Path) -> None:
    """New project "Source must be an absolute path" bug: a path pasted with the quotes of
    Windows' Copy as path, a CRLF line's \\r or blanks around it is that path, and blank lines
    are left out; a relative path is still refused. The projects folder setting reads it so too."""
    footage = tmp_path / "footage"
    footage.mkdir()
    a, b = footage / "a.mp4", footage / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "T", "sources": [f'"{a}"', f" {b}\t\r", ""]})
    assert res.status_code == 200, res.text
    assert ui.read_meta(ui.project_dir(res.json()["id"]))["inputs"] == [str(a), str(b)]
    res = client.post("/api/projects", json={"name": "T", "source": f'"{a}"'})
    assert res.status_code == 200, res.text
    assert ui.read_meta(ui.project_dir(res.json()["id"]))["source"] == str(a)
    res = client.post("/api/projects", json={"name": "T", "sources": ['"footage/a.mp4"']})
    assert res.status_code == 400
    assert len(client.get("/api/projects").json()["projects"]) == 2
    res = client.put("/api/settings", json={"projects_dir": f'"{tmp_path / "elsewhere"}"'})
    assert res.status_code == 200, res.text
    assert res.json()["projects_dir"] == str(tmp_path / "elsewhere")


def test_create_project_expands_home_in_sources(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source or projects folder typed from ``~`` (quoted too) is the path in the home folder,
    stored expanded; a relative path is still refused."""
    home = tmp_path / "home"
    (home / "Videos").mkdir(parents=True)
    a, b = home / "Videos" / "a.mp4", home / "Videos" / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    quoted = f'"~{os.sep}Videos{os.sep}b.mp4"'  # a backslash is a separator on Windows alone
    res = client.post("/api/projects", json={"name": "T", "sources": ["~/Videos/a.mp4", quoted]})
    assert res.status_code == 200, res.text
    assert ui.read_meta(ui.project_dir(res.json()["id"]))["inputs"] == [str(a), str(b)]
    res = client.post("/api/projects", json={"name": "T", "source": '"~/Videos"'})
    assert res.status_code == 200, res.text
    assert Path(ui.read_meta(ui.project_dir(res.json()["id"]))["source"]) == home / "Videos"
    res = client.post("/api/projects", json={"name": "T", "sources": ["Videos/a.mp4"]})
    assert res.status_code == 400
    assert "source must be an absolute path" in res.json()["detail"]
    res = client.put("/api/settings", json={"projects_dir": "~/projects"})
    assert res.status_code == 200, res.text
    assert res.json()["projects_dir"] == str(home / "projects")
    assert (home / "projects").is_dir()


def test_360_project_lists_one_lrv_per_usable_clip_before_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in (*X3_CLIPS, "VID_20260101_110000_10_002.insv", "LRV_20260101_110000_11_002.lrv"):
        (folder / name).write_bytes(b"")  # clip 002 has no _00_ lens: not usable
    res = client.post("/api/projects", json={"name": "T", "source": str(folder)})
    p = ui.project_dir(res.json()["id"])
    assert [Path(e["path"]).name for e in ui.source_entries(p, None)] == [X3_LRV]


def test_360_file_pick_takes_its_recording_when_another_has_its_number(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A207: one picked file of a 029 two recordings share gives that recording alone, its
    analysis is accepted and it lists its own proxy; a project with the bare number still
    lists its clip."""
    folder = tmp_path / "footage"
    folder.mkdir()
    names = [
        "VID_20240831_194807_00_029.insv", "LRV_20240831_194807_01_029.lrv",
        "VID_20260122_132929_00_029.insv", "LRV_20260122_132929_11_029.lrv",
    ]  # fmt: skip
    for name in names:
        (folder / name).write_bytes(b"")
    res = client.post("/api/projects", json={"name": "T", "source": str(folder / names[2])})
    assert res.status_code == 200, res.text
    p = ui.project_dir(res.json()["id"])
    assert ui.read_meta(p)["clips"] == ["20260122_132929_029"]
    assert [Path(e["path"]).name for e in ui.source_entries(p, None)] == [names[3]]
    patch_ui(monkeypatch, "cutpoint_cmd", lambda *a: ["cutpoint", *a])
    seen = JobSpy(monkeypatch, check=False).args
    res = client.post(f"/api/projects/{res.json()['id']}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert seen[0][-2:] == ["--only", "20260122_132929_029"]
    cache.save_json(p / ui.META_FILE, {**ui.read_meta(p), "clips": ["029"]})  # before A207
    assert [Path(e["path"]).name for e in ui.source_entries(p, None)] == [names[3]]
    seen.clear()
    res = client.post(f"/api/projects/{p.name}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert seen[0][-2:] == ["--only", "20260122_132929_029"]


def test_list_projects_shows_ui_360_projects_but_hides_command_line_ones(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    pid = client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    cache.save_json(ui.project_dir(pid) / "pano.json", {"clips": {}})  # as after the scan
    cli_only = ui.projects_dir() / "cli-360"
    cli_only.mkdir()
    cache.save_json(cli_only / "pano.json", {"clips": {}})
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == [pid]


@pytest.mark.parametrize("only", [None, "20260101_100000_001"], ids=["folder", "file"])
def test_360_analyze_job_runs_the_360_command_with_a_plan(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, only: str | None
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    source = folder / X3_00 if only else folder
    pid = client.post("/api/projects", json={"name": "T", "source": str(source)}).json()["id"]
    patch_ui(monkeypatch, "cutpoint_cmd", lambda *a: ["cutpoint", *a])
    seen = JobSpy(monkeypatch, check=False).args
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    p = ui.project_dir(pid)
    want = ["cutpoint", "360", "analyze", str(p), "--scan", str(folder), "--style", "action-cam"]
    assert seen == [want + (["--only", only] if only else [])]
    assert "--vision" not in seen[0] and "--no-vision" not in seen[0]
    # the Analyze options are analyze flags 360 analyze does not take: refused, not dropped
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze", "no_scenes": True})
    assert res.status_code == 400 and "not available for 360 projects" in res.json()["detail"]
    assert len(seen) == 1


def test_360_analyze_job_view_carries_the_step_plan(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    pid = client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    # No real interpreter: spawn starts jobs at below-normal priority, and on a busy machine
    # one took up to ~15 s to exit, close to the old 20 s wait. The plan comes from the real Job.
    FakeProc().fake_spawn(monkeypatch)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert [s["step"] for s in res.json()["progress"]["plan"]] == [
        "scan", "views", "analyze", "events",
    ]  # fmt: skip
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)


class StitchExport:
    """A 360 project with one recorded clip and one moment, and fake job processes.

    ``spawn`` records each command; the ``360 stitch`` one exits ``stitch_code`` (after
    ``hold`` is set, when given), the ``clips`` one writes its manifest. ``sdk_status`` and the
    free space are mocked: the real SDK never runs.
    """

    def __init__(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        sdk: dict | None = None,
        free: float | None = 500.0,
        nvidia: bool = True,
    ) -> None:
        folder = tmp_path / "footage"
        folder.mkdir()
        for name in X3_CLIPS:
            (folder / name).write_bytes(b"")
        res = client.post("/api/projects", json={"name": "T", "source": str(folder)})
        self.client, self.pid = client, res.json()["id"]
        self.p = ui.project_dir(self.pid)
        lrv = str(folder / X3_LRV)
        self.clip = x3clip(lrv=lrv, insv=[str(folder / X3_00), str(folder / X3_10)], duration=60)
        pano_project.save(self.p, [self.clip])
        cand = {"id": "h001", "source": lrv, "start": 1.0, "end": 5.0, "score": 80, "label": "a"}
        hl = {
            "sources": [{"path": lrv, "duration": 60.0}],
            "candidates": [cand],
            "style": "action-cam",
        }
        cache.save_json(self.p / "highlights.json", hl)
        self.segment = {k: cand[k] for k in ("id", "source", "start", "end", "label")}
        self.seen: list[list[str]] = []
        self.stitch_code = 0
        self.stitch_lines = ["10:00:00 INFO clip 001: ok"]
        self.clips_code = 0
        self.clips_lines: list[str] = []
        self.hold: threading.Event | None = None
        self.stitching = threading.Event()
        patch_ui(monkeypatch, "cutpoint_cmd", lambda *a: ["cutpoint", *a])
        patch_ui(monkeypatch, "spawn", self.spawn)
        monkeypatch.setattr(procs, "kill_tree", lambda proc, *a, **k: proc.release.set())
        status = {"found": True, "runs": True} if sdk is None else sdk
        monkeypatch.setattr(pano_stitch, "sdk_status", lambda: status)
        monkeypatch.setattr(pano_stitch, "free_gb", lambda _folder: free)
        monkeypatch.setattr(capabilities, "nvidia_present", lambda: nvidia)

    def spawn(self, args: list[str], **_kw: object) -> FakeProc:
        self.seen.append(args)
        if args[1:3] == ["360", "stitch"]:
            proc = FakeProc(self.stitch_lines, self.stitch_code)
            self.stitching.set()
            if self.hold is None:
                proc.release.set()
            else:  # until cancelled
                self.hold = proc.release
            return proc
        out = Path(args[args.index("--out") + 1])
        out.mkdir()
        cache.save_json(out / "manifest.json", {"clips": []})
        proc = FakeProc(self.clips_lines, self.clips_code)
        proc.release.set()
        return proc

    def export(self, body: dict | None = None) -> dict:
        """Start an export (the sequence's, unless ``body`` says otherwise); the job's view."""
        body = body or {}
        if body.get("picks") != "auto":
            res = self.client.put(
                f"/api/projects/{self.pid}/review", json={"segments": [self.segment]}
            )
            assert res.status_code == 200, res.text
        res = self.client.post(f"/api/projects/{self.pid}/jobs", json={"action": "clips", **body})
        assert res.status_code == 200, res.text
        job = self.client.app.state.jobs.get(res.json()["id"])
        if self.hold is None:
            assert job.done.wait(10)
        return job

    def commands(self) -> list[list[str]]:
        return [a[1:3] for a in self.seen]


def test_a_360_export_stitches_the_sequence_recordings_first(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    view = run.export().view()
    assert view["status"] == "completed", view
    stitch, clips = run.seen
    (seg_file,) = run.p.glob("ui/segments-*.json")
    assert stitch == ["cutpoint", "360", "stitch", str(run.p), "--segments", str(seg_file)]
    assert clips[1:3] == ["clips", str(run.p)]
    assert any("stitching 1 recording(s)" in ln and "500 GB free" in ln for ln in view["logs"])


def test_a_360_export_of_automatic_picks_stitches_their_clips(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    assert run.export({"picks": "auto", "ids": ["h001"]}).view()["status"] == "completed"
    assert run.seen[0] == ["cutpoint", "360", "stitch", str(run.p), "--clips", "001"]
    assert run.commands()[1] == ["clips", str(run.p)]


@pytest.mark.parametrize(
    ("pano", "sdk", "said"),
    [
        ({"auto_stitch": False}, None, "pano.auto_stitch is off"),
        ({}, {"found": False, "runs": False}, "SDK was not found"),
        ({}, {"found": True, "runs": False}, "SDK does not run"),
    ],
    ids=["auto-stitch-off", "no-sdk", "sdk-does-not-run"],
)
def test_a_360_export_without_auto_stitch_or_a_working_sdk_exports_the_preview(
    client: TestClient,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pano: dict,
    sdk: dict | None,
    said: str,
) -> None:
    set_config(home, pano=pano)
    run = StitchExport(client, tmp_path, monkeypatch, sdk=sdk)
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["clips", str(run.p)]]
    assert any(said in ln and "360 preview (not stitched)" in ln for ln in view["logs"])


def test_a_360_export_without_an_nvidia_gpu_skips_the_stitch_with_a_warning(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch, nvidia=False)
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["clips", str(run.p)]]
    assert any("WARNING no NVIDIA GPU; exporting the 360 preview" in ln for ln in view["logs"])


def test_a_360_export_skips_the_stitch_with_a_warning_when_space_is_short(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch, free=pano_stitch.MIN_FREE_GB + 0.1)
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["clips", str(run.p)]]
    assert any("WARNING not enough free space" in ln for ln in view["logs"])


def test_a_360_export_of_stitched_recordings_does_not_stitch_again(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    write(pano_project.sdk_output(run.clip), b"stitched")
    assert run.export().view()["status"] == "completed"
    assert run.commands() == [["clips", str(run.p)]]


def test_a_failed_stitch_still_exports_with_a_warning(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_code = 1
    view = run.export().view()
    assert view["status"] == "completed" and "error" not in view, view
    assert run.commands() == [["360", "stitch"], ["clips", str(run.p)]]
    assert any("WARNING the step before clips failed (exit code 1)" in ln for ln in view["logs"])


def test_a_stitch_that_exits_0_after_an_error_warns_and_still_exports(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``360 stitch`` exits 0 when the SDK failed on a recording; its ERROR line says so."""
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_lines = ["10:00:00 ERROR clip 001: stitch FAILED: x"]
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["360", "stitch"], ["clips", str(run.p)]]
    want = "WARNING the step before clips reported errors: those recordings export from the 360"
    assert any(want in ln for ln in view["logs"])


@pytest.mark.parametrize(
    ("clips_lines", "error"),
    [([], "exit code 1"), (["10:00:00 ERROR own"], "own")],
    ids=["no-error-line", "own-error-line"],
)
def test_a_failed_export_after_a_failed_stitch_reports_its_own_error(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clips_lines: list[str],
    error: str,
) -> None:
    """The job's error comes from the clips step, never from the stitch's ERROR lines."""
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_lines, run.stitch_code = ["10:00:00 ERROR clip 001: stitch FAILED: x"], 1
    run.clips_lines, run.clips_code = clips_lines, 1
    view = run.export().view()
    assert view["status"] == "failed", view
    assert view["error"] == error


def test_cancelling_a_360_export_while_it_stitches_stops_both_steps(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_code = 130
    run.hold = threading.Event()
    job = run.export()
    assert run.stitching.wait(10)
    client.post(f"/api/jobs/{job.id}/cancel")
    assert job.done.wait(10)
    assert job.view()["status"] == "cancelled"
    assert run.commands() == [["360", "stitch"]]  # clips never started


def test_a_flat_export_never_stitches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    monkeypatch.setattr(
        pano_stitch, "sdk_status", lambda: pytest.fail("a flat export asked for the SDK")
    )
    FakeProc().fake_spawn(monkeypatch)
    res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)


def test_create_project_skips_unreadable_folder_entries(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry whose is_file() raises PermissionError must not make a good folder fail."""
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    (folder / "locked").mkdir()
    real = Path.is_file

    def is_file(self: Path) -> bool:
        if self.name == "locked":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    res = client.post("/api/projects", json={"name": "Trip", "source": str(folder)})
    assert res.status_code == 200, res.text


class DirectorPreviews:
    """A UI 360 project whose analysis cached camera paths (#517): ``h001`` current and flagged
    (sky, whip pans, failed QA), ``h002`` stale (planned with other settings), ``h003`` none.

    The cache entries are written in ``cutpoint.pano.candidates``' own layout and key, so its
    preview API (``candidate_flags``, ``preview_path``) runs for real. Only the render is faked
    (``pano.render.render_segment``): it reports ``progress`` 0.5, writes through
    ``atomic_output`` as the real one does and, with ``hold``, blocks like ffmpeg until its child
    scope is stopped.
    """

    PLANNED = (0.0, 15.0)

    def __init__(self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        PC = pano_candidates
        folder = tmp_path / "footage"
        folder.mkdir()
        for name in X3_CLIPS:
            (folder / name).write_bytes(b"")
        res = client.post("/api/projects", json={"name": "T", "source": str(folder)})
        self.client, self.pid = client, res.json()["id"]
        self.p = ui.project_dir(self.pid)
        self.lrv = str(folder / X3_LRV)
        clip = x3clip(
            lrv=self.lrv, insv=[str(folder / X3_00), str(folder / X3_10)], fps=30.0, duration=60
        )
        self.clip = clip
        pano_project.save(self.p, [clip])
        # h003 starts off a frame: its level forward view must still show its first frame
        spans = {"h001": (5.0, 10.0), "h002": (20.0, 25.0), "h003": (40.01, 45.0)}
        cands = [
            {"id": c, "source": self.lrv, "start": a, "end": b, "score": 50, "label": c}
            for c, (a, b) in spans.items()
        ]
        hl = {"sources": [{"path": self.lrv, "duration": 60.0}], "candidates": cands}
        cache.save_json(self.p / "highlights.json", {**hl, "style": "action-cam"})
        folder_c = self.p / PC.CACHE_DIR
        folder_c.mkdir()
        diag = {
            "seconds": 5.0,
            "quality": {"score": 40.0, "pass": False, "fails": ["edge_cut 27%"]},
            "sky": 0.5,
            "peak_yaw_rate": 80.0,
            "cuts": 0,
            "cuts_per_min": 0.0,
        }
        for cid in ("h001", "h002"):
            a, b = spans[cid]
            key = PC.path_key(clip, a, b, None, None)
            if cid == "h002":
                key["params"] = {**key["params"], "w_forward": -1.0}  # other settings: stale
            path = {
                "clip": "001",
                "source": self.lrv,
                "segment": {"start": a, "end": b},
                "planned": {"start": self.PLANNED[0] + a - 5.0, "end": self.PLANNED[1] + a - 5.0},
                "vertical": False,
                "keys": [{"t": a, "yaw": 0.0, "pitch": 0.0, "fov": 90.0, "target": "forward"}],
            }
            cache.save_json(
                folder_c / PC.slot_name(clip, a, b),
                {
                    "version": PC.CACHE_VERSION,
                    "key": key,
                    "candidate": {"id": cid, "source": self.lrv, "start": a, "end": b},
                    "diagnostics": diag if cid == "h001" else {"seconds": 5.0},
                    "path": path,
                },
            )
        self.renders: list[dict] = []
        self.hold: threading.Event | None = None
        self.started = threading.Event()
        self.fail: Exception | None = None

        def render_segment(clip, path, start, end, out, **kw):
            self.renders.append(
                {"clip": clip, "path": path, "start": start, "end": end, "out": out}
            )
            self.renders[-1].update(kw)
            (kw["work_dir"] / f"{out.stem}.cmd").write_text("# cmd\n", encoding="utf-8")
            with folders.atomic_output(out) as part:
                part.write_bytes(b"partial")
                kw["progress"](0.5)
                self.started.set()
                scope = ffmpeg_utils._SCOPE.current
                while self.hold is not None and not self.hold.is_set():
                    if scope.wait_cancelled(0.02):  # what run() does once its ffmpeg is killed
                        raise KeyboardInterrupt
                if self.fail is not None:
                    raise self.fail
                part.write_bytes(b"director view")
            kw["progress"](1.0)
            return {"mode": "proxy", "seconds": 0.0}

        monkeypatch.setattr(pano_render, "render_segment", render_segment)

    def candidate(self, cid: str) -> dict:
        res = self.client.get(f"/api/projects/{self.pid}")
        assert res.status_code == 200, res.text
        return next(c for c in res.json()["highlights"]["candidates"] if c["cli_id"] == cid)

    def ask(self, cid: str, cancel: bool = False):
        return self.client.post(
            f"/api/projects/{self.pid}/director/{cid}" + ("/cancel" if cancel else "")
        )

    def wait(self, cid: str, *states: str) -> dict:
        """The candidate's preview once its state is one of ``states``."""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            pv = self.candidate(cid)["pano"]["preview"]
            if pv is not None and pv["state"] in states:
                return pv
            time.sleep(0.02)
        pytest.fail(f"{cid} never reached {states}: {pv}")

    def folder(self) -> list[str]:
        d = self.p / ui.UI_DIR / "pano"
        return sorted(f.name for f in d.iterdir()) if d.is_dir() else []


def test_360_moments_carry_camera_path_flags_and_reasons(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#517: each 360 moment lists the flags of its cached camera path, each with the reason
    and its measure; one without a cached path has none."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    pano = run.candidate("h001")["pano"]
    assert pano["flags"] == ["sky", "whip", "qa_fail"]
    assert [r["text"] for r in pano["reasons"]] == [
        "looks at the sky (50% of the moment)",
        "whip pans (up to 80 deg/s)",
        "fails the geometric direction QA: edge_cut 27%",
    ]
    assert pano["preview"] is None  # nothing is rendered before the page asks
    h002 = run.candidate("h002")["pano"]
    assert {**h002, "path": None} == {"flags": [], "reasons": [], "path": None, "preview": None}
    assert run.candidate("h003")["pano"]["flags"] == []
    assert run.renders == []


def test_a_non_360_project_has_no_camera_path_data_or_director_route(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    cand = {"id": "h001", "source": str(folder / "a.mp4"), "start": 1.0, "end": 2.0, "score": 1}
    cache.save_json(ui.project_dir(pid) / "highlights.json", {"candidates": [cand]})
    got = client.get(f"/api/projects/{pid}").json()["highlights"]["candidates"][0]
    assert "pano" not in got
    assert client.post(f"/api/projects/{pid}/director/h001").status_code == 404


def test_a_director_preview_renders_in_the_background_and_is_reused(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current path renders once (640x360 from the .lrv, its planned span) into ui/pano and is
    served from there; asking again, or after a restart, renders nothing."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    res = run.ask("h001")
    assert res.status_code == 200, res.text
    assert res.json()["pano"]["preview"]["state"] in ("queued", "rendering", "ready")
    pv = run.wait("h001", "ready")
    assert (pv["lo"], pv["hi"]) == run.PLANNED
    assert len(run.renders) == 1
    r = run.renders[0]
    assert (r["start"], r["end"]) == run.PLANNED and r["clip"] == run.clip
    assert (r["fmt"].width, r["fmt"].height) == (640, 360) and r["cmd_every"] == 1
    assert r["src"].inputs == (run.lrv,) and r["src"].projection == "dfisheye"
    assert r["out"].parent == run.p / ui.UI_DIR / "pano"
    assert run.folder() == [r["out"].name]  # no partial, no sendcmd file left
    sid = run.client.get(f"/api/projects/{run.pid}").json()["sources"][0]["id"]
    assert f"/sources/{sid}/director/" in pv["url"]
    media = run.client.get(pv["url"].replace("http://127.0.0.1", ""))
    assert media.status_code == 200 and media.content == b"director view"
    assert run.ask("h001").json()["pano"]["preview"]["state"] == "ready"
    assert len(run.renders) == 1

    app = ui.create_app(PORT)  # a restart: the file on disk is reused once the path checks out
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c2:
        c2.headers["X-Cutpoint-Token"] = app.state.token
        listed = c2.get(f"/api/projects/{run.pid}").json()["highlights"]["candidates"][0]
        assert listed["pano"]["preview"] is None  # not checked current yet this run
        c2.post(f"/api/projects/{run.pid}/director/h001")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            got = c2.get(f"/api/projects/{run.pid}").json()["highlights"]["candidates"][0]
            if (got["pano"]["preview"] or {}).get("state") == "ready":
                break
            time.sleep(0.02)
        assert got["pano"]["preview"]["state"] == "ready"
    assert len(run.renders) == 1


def test_two_apps_in_one_process_share_no_planners_or_login_states(tmp_path: Path) -> None:
    """The routes read the state of the app that served them, not a module-level dict."""
    a, b = ui.create_app(PORT), ui.create_app(PORT)
    assert a.state.planners is not b.state.planners
    assert a.state.login_states is not b.state.login_states
    proj = tmp_path / "proj"
    assistant_routes.edge_planner(a, proj)
    assert proj in a.state.planners and b.state.planners == {}

    provider = cliproxy.PROVIDERS[0]
    a.state.login_states[provider] = {"state": "done", "error": ""}
    seen = {}
    for name, app in (("a", a), ("b", b)):
        with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
            c.headers["X-Cutpoint-Token"] = app.state.token
            res = c.get("/api/assistant/login/status", params={"provider": provider})
            assert res.status_code == 200, res.text
            seen[name] = res.json()["state"]
    assert seen == {"a": "done", "b": "idle"}


def test_two_apps_in_one_process_serve_only_their_own_media_urls(home: Path) -> None:
    """A media URL carries the media token of the app that listed it (app.state.media_token),
    so it plays from that app and the other app's media routes refuse it."""
    proj = ui.projects_dir() / "proj"
    write(proj / "ui" / "meta.json", json.dumps({"name": "proj"}))
    write(proj / "final.mp4", b"video")
    a, b = ui.create_app(PORT), ui.create_app(PORT)
    urls = {}
    for name, app in (("a", a), ("b", b)):
        with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
            c.headers["X-Cutpoint-Token"] = app.state.token
            res = c.get("/api/projects/proj")
            assert res.status_code == 200, res.text
            urls[name] = res.json()["outputs"][0]["url"]
    assert urls["a"] != urls["b"]
    for name, app in (("a", a), ("b", b)):
        with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:  # the URL's token only
            for listed_by, url in urls.items():
                res = c.get(url)
                assert res.status_code == (200 if listed_by == name else 403), (name, url)


@pytest.mark.parametrize("path,body", [("target", {"seconds": 30}), ("review", {"segments": []})])
def test_two_apps_in_one_process_hold_their_own_review_lock(
    home: Path, path: str, body: dict
) -> None:
    """Saving the sequence or the target takes the review lock of the app that serves it
    (app.state.review_lock): a save waiting in one app never holds up the other's."""
    write(ui.projects_dir() / "proj" / "ui" / "meta.json", json.dumps({"name": "proj"}))
    a, b = ui.create_app(PORT), ui.create_app(PORT)

    def put_save(app) -> int:
        with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
            c.headers["X-Cutpoint-Token"] = app.state.token
            return c.put(f"/api/projects/proj/{path}", json=body).status_code

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        with a.state.review_lock:
            assert pool.submit(put_save, b).result(timeout=30) == 200
            held = pool.submit(put_save, a)
            with pytest.raises(concurrent.futures.TimeoutError):
                held.result(timeout=0.5)  # waits for its own app's lock
        assert held.result(timeout=30) == 200


def test_a_stale_or_missing_camera_path_previews_the_library_fallback(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#517: a moment whose cached path is stale or missing previews the path
    ``candidates.preview_path`` gives (here, as these clips cannot be planned, a level forward
    view), with its reason, instead of the raw fisheye. A level forward view is planned with the
    director's margin (``PATH_MARGIN_S``) around the moment, so its render starts on the frame
    at or before the moment's start less that margin (rounded up, the page played the raw
    fisheye there)."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    for cid in ("h002", "h003"):
        run.ask(cid)
        pv = run.wait(cid, "ready")
        assert pv["kind"] == "level_forward" and pv["reason"]
        assert run.renders[-1]["path"]["keys"][0]["target"] == "forward"
    want_lo = 40.01 - pano_director.PATH_MARGIN_S
    assert pv["lo"] <= want_lo and pv["lo"] == round(math.floor(want_lo * 30) / 30, 6)
    assert round(run.renders[-1]["start"], 6) == pv["lo"]
    assert run.ask("nope").json() == {"pano": None}


def test_cancelling_a_director_preview_removes_its_partial_render(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    run.hold = threading.Event()
    run.ask("h001")
    assert run.started.wait(10)
    pv = run.wait("h001", "rendering")
    assert (pv["seconds"], pv["progress"]) == (15.0, 0.5)  # render_segment's progress
    assert any(n.endswith(".part.mp4") for n in run.folder())
    pv = run.ask("h001", cancel=True).json()["pano"]["preview"]
    assert pv == {"state": "cancelled", "reason": ""}
    deadline = time.monotonic() + 10
    while run.folder() and time.monotonic() < deadline:  # the render thread cleans up
        time.sleep(0.02)
    assert run.folder() == []  # no .part, no .mp4, no .cmd
    assert run.candidate("h001")["pano"]["preview"]["state"] == "cancelled"  # no auto re-render
    run.hold = None
    run.started.clear()
    run.ask("h001")  # "Render director view"
    assert run.wait("h001", "ready")
    assert len(run.renders) == 2


def test_a_failed_director_preview_reports_why_and_leaves_no_file(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    run.fail = errors.FFmpegError("ffmpeg failed (exit 1):\nInvalid data found")
    run.ask("h001")
    pv = run.wait("h001", "failed")
    assert pv == {"state": "failed", "reason": "Invalid data found"}
    assert run.folder() == []


def test_leftover_director_previews_of_old_paths_are_pruned(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    folder = run.p / ui.UI_DIR / "pano"
    folder.mkdir(parents=True)
    old = time.time() - ui.PROXY_PRUNE_GRACE_S - 60
    names = ["h099-0123456789abcdef.mp4", ".h001-0123456789abcdef.part.mp4", "notes.txt"]
    for name in names:
        (folder / name).write_bytes(b"x")
        os.utime(folder / name, (old, old))
    (folder / "h009-fedcba9876543210.mp4").write_bytes(b"x")  # recent: another run may serve it
    # review F1: a moment's newest preview stays (a path planned now is named only once planned)
    newest = folder / "h003-aaaaaaaaaaaaaaaa.mp4"
    newest.write_bytes(b"x")
    os.utime(newest, (old + 30, old + 30))
    older = folder / "h003-bbbbbbbbbbbbbbbb.mp4"
    older.write_bytes(b"x")
    os.utime(older, (old, old))
    run.candidate("h001")
    assert run.folder() == [newest.name, "h009-fedcba9876543210.mp4", "notes.txt"]


def test_a_replanned_camera_path_has_a_new_identity(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F1: the page asks again per path identity, so "Analyze again" (a new plan in the
    cache) must change ``path``; the old ready preview no longer counts."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    before = run.candidate("h001")["pano"]["path"]
    assert isinstance(before, str) and before.startswith("h001-")
    run.ask("h001")
    run.wait("h001", "ready")
    slot = run.p / pano_candidates.CACHE_DIR / pano_candidates.slot_name(run.clip, 5.0, 10.0)
    entry = cache.load_json(slot)
    entry["path"]["keys"][0]["yaw"] = 30.0  # re-planned
    cache.save_json(slot, entry)
    later = time.time() + 5
    os.utime(slot, (later, later))
    pano = run.candidate("h001")["pano"]
    assert pano["path"] != before and pano["path"].startswith("h001-")
    assert pano["preview"] is None  # the page asks for the new path's preview
    assert run.ask("h001").json()["pano"]["preview"]["state"] in ("queued", "rendering", "ready")
    run.wait("h001", "ready")
    assert len(run.renders) == 2


def test_a_director_preview_is_only_for_a_project_source(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F4: a candidate whose source is not one of the project's gets ``{"pano": null}``
    and renders nothing; the media route serves only the listed preview of that source."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    run.ask("h001")
    pv = run.wait("h001", "ready")
    stray = run.p / ui.UI_DIR / "pano" / "h002-0123456789abcdef.mp4"
    stray.write_bytes(b"x")  # a preview-like file no listed candidate names
    url = pv["url"].replace("http://127.0.0.1", "")
    head, tail = url.split("/director/")
    assert run.client.get(url).status_code == 200
    stray_url = f"{head}/director/{stray.stem}?{tail.split('?')[1]}"
    assert run.client.get(stray_url).status_code == 404
    hl = cache.load_json(run.p / "highlights.json")
    hl["candidates"][0]["source"] = str(tmp_path / "elsewhere" / X3_LRV)
    cache.save_json(run.p / "highlights.json", hl)
    assert run.ask("h001").json() == {"pano": None}
    assert len(run.renders) == 1


class DirectorReads:
    """Counts what the director route reads for a request (#1029): ``highlights.json`` loads and
    candidate states built (``listings``: one per candidate of each rebuild)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.highlights = self.listings = 0
        real_load, real_state = cache.load_json, ui.PanoPreviews._state

        def load_json(path: Any, *a: Any, **kw: Any) -> Any:
            if Path(path).name == "highlights.json":
                self.highlights += 1
            return real_load(path, *a, **kw)

        def state(previews: ui.PanoPreviews, project: Path, cid: str, info: dict) -> Any:
            self.listings += 1  # a rebuild of the ready previews asks each candidate's state
            return real_state(previews, project, cid, info)

        monkeypatch.setattr(cache, "load_json", load_json)
        monkeypatch.setattr(ui.PanoPreviews, "_state", state)

    def reset(self) -> None:
        self.highlights = self.listings = 0


def ready_director(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DirectorPreviews, str]:
    """A 360 project with ``h001``'s preview ready, and that preview's URL path."""
    patch_ui(monkeypatch, "MEDIA_LOOKUP_S", 3600.0)  # the lookups stay kept however slow
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    run.ask("h001")
    pv = run.wait("h001", "ready")
    return run, pv["url"].replace("http://127.0.0.1", "")


def test_repeated_director_requests_read_neither_highlights_nor_the_listing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1029: an unchanged project answers the many Range requests of one preview from memory."""
    _run, url = ready_director(client, tmp_path, monkeypatch)
    assert client.get(url).status_code == 200  # builds the kept answer
    reads = DirectorReads(monkeypatch)
    for i in range(9):
        res = client.get(url, headers={"Range": f"bytes={i}-{i + 3}"})
        assert (res.status_code, res.content) == (206, b"director view"[i : i + 4])
    assert (reads.highlights, reads.listings) == (0, 0)


def test_a_rewritten_highlights_file_is_seen_by_the_director_route(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, url = ready_director(client, tmp_path, monkeypatch)
    assert client.get(url).status_code == 200
    reads = DirectorReads(monkeypatch)
    hl = cache.load_json(run.p / "highlights.json")
    hl["candidates"][0]["source"] = str(tmp_path / "elsewhere" / X3_LRV)  # as analyze or Undo do
    cache.save_json(run.p / "highlights.json", hl)
    reads.reset()
    assert client.get(url).status_code == 404  # no longer a moment of this source
    assert reads.highlights >= 1 and reads.listings >= 1
    hl["candidates"][0]["source"] = run.lrv  # rewritten back
    cache.save_json(run.p / "highlights.json", hl)
    assert client.get(url).status_code == 200


def test_a_preview_removed_or_added_is_seen_by_the_director_route(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, url = ready_director(client, tmp_path, monkeypatch)
    assert client.get(url).status_code == 200
    (run.p / ui.UI_DIR / "pano").joinpath(
        url.split("/director/")[1].split("?")[0] + ".mp4"
    ).unlink()
    assert client.get(url).status_code == 404  # removed
    run.ask("h001")  # rendered again
    run.wait("h001", "ready")
    assert client.get(url).status_code == 200  # added
    reads = DirectorReads(monkeypatch)
    assert client.get(url).status_code == 200
    assert client.get(url).status_code == 200
    assert reads.listings <= 3  # at most one rebuild (3 candidates) of what the render changed


def test_a_changed_render_setting_or_camera_path_stops_serving_the_preview(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    run, url = ready_director(client, tmp_path, monkeypatch)
    assert client.get(url).status_code == 200
    set_config(home, pano={"level_strength": 0.5})
    assert client.get(url).status_code == 404  # a new identity: no re-render needed to see it
    assert len(run.renders) == 1
    set_config(home)
    assert client.get(url).status_code == 200
    slot = run.p / pano_candidates.CACHE_DIR / pano_candidates.slot_name(run.clip, 5.0, 10.0)
    entry = cache.load_json(slot)
    entry["path"]["keys"][0]["yaw"] = 30.0  # re-planned
    cache.save_json(slot, entry)
    assert client.get(url).status_code == 404
    assert len(run.renders) == 1


def test_director_answers_are_per_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, url = ready_director(client, tmp_path, monkeypatch)
    assert client.get(url).status_code == 200
    (tmp_path / "two").mkdir()
    pid2, _ = make_project(client, tmp_path / "two", {"a.mp4": b"x"})
    sid2 = client.get(f"/api/projects/{pid2}").json()["sources"][0]["id"]
    name = url.split("/director/")[1]
    other = f"/api/projects/{pid2}/sources/{sid2}/director/{name}"
    assert client.get(other).status_code == 404  # the first project's preview is not served here
    assert client.get(url).status_code == 200
    previews: ui.PanoPreviews = client.app.state.previews
    assert previews.ready(ui.project_dir(pid2)) == {}
    assert set(previews.ready(run.p)) == {"h001"}
    assert set(previews.ready_memo) >= {run.p, ui.project_dir(pid2)}


def test_a_changed_imu_offset_or_render_format_renders_a_fresh_preview(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """Review of #541: the library's path ``key`` covers only the path, but a preview is also
    levelled with ``pano.imu_offset_s`` and rendered in the UI's format; changing either must
    give a new preview (a new identity the page asks again for, a new render), not the old
    file."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    names = []
    for change in (
        lambda: None,
        lambda: set_config(home, pano={"imu_offset_s": 0.25}),
        lambda: patch_ui(monkeypatch, "PANO_PREVIEW_W", 480),
    ):
        change()
        assert run.candidate("h001")["pano"]["preview"] is None  # nothing ready for this one
        run.ask("h001")
        names.append(run.wait("h001", "ready")["url"].split("/director/")[1].split("?")[0])
    assert len(run.renders) == 3 and len(set(names)) == 3
    assert run.renders[-1]["fmt"].width == 480


def test_a_director_preview_is_named_by_the_library_preview_key(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """A preview's file is ``<candidate>-<candidates.preview_key>`` of the path it renders, its
    16:9 format and the pano settings, so a changed render setting names another file."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    names = []
    for level in (None, 0.5):
        if level is not None:
            set_config(home, pano={"level_strength": level})
        run.ask("h001")
        run.wait("h001", "ready")
        r = run.renders[-1]
        key = pano_candidates.preview_key(
            r["path"], vertical=False, fmt=r["fmt"], settings=pano_project.pano_config()
        )
        assert r["out"].name == f"h001-{key}.mp4"
        assert ui.PanoPreviews().entries(run.p)["h001"]["name"] == r["out"].stem  # prune keeps it
        names.append(r["out"].name)
    assert len(run.renders) == 2 and names[0] != names[1]


def test_a_spinning_camera_path_reaches_the_moment_badges(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#541 added the ``spin`` flag (a director that keeps turning): the listing passes it on
    from ``candidate_flags``, with its reason, for the card's badge and tooltip."""
    run = DirectorPreviews(client, tmp_path, monkeypatch)
    slot = run.p / pano_candidates.CACHE_DIR / pano_candidates.slot_name(run.clip, 20.0, 25.0)
    entry = cache.load_json(slot)
    entry["diagnostics"] |= {"net_turns": 1.5, "yaw_travel": 540.0, "yaw_travel_per_min": 6480.0}
    cache.save_json(slot, entry)
    pano = run.candidate("h002")["pano"]
    assert pano["flags"] == ["spin"]
    assert pano["reasons"][0]["flag"] == "spin"
    assert pano["reasons"][0]["text"].startswith(pano_candidates.FLAGS["spin"])


def test_the_spin_reason_names_a_short_moment_s_own_length() -> None:
    """#547: under SPIN_WINDOW_S the turns are counted over the whole moment, so the reason
    names its length, not the 30 s window."""
    diag = {"net_turns": 0.42, "yaw_travel_per_min": 1500.0}
    short = ui._pano_reasons(["spin"], {}, diag | {"seconds": 6.0})[0]["text"]
    assert "0.42 turns one way in 6 s, 1500 deg of pan per minute" in short
    for d in (diag | {"seconds": 45.0}, diag):  # a long moment, or a record without its length
        text = ui._pano_reasons(["spin"], {}, d)[0]["text"]
        assert f"one way in {pano_candidates.SPIN_WINDOW_S:g} s" in text


def test_find_videos_never_returns_raw_360_files(tmp_path: Path) -> None:
    """analyze must not read dual-fisheye .insv or .lrv proxies as flat video."""
    for name in (X3_00, X3_10, X3_LRV, "a.mp4"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_videos([tmp_path])] == ["a.mp4"]


@pytest.mark.parametrize("ext", MPEG_EXTS)  # the list find_videos is tested with
def test_source_problem_accepts_mpeg_transport_and_program_streams(
    tmp_path: Path, ext: str
) -> None:
    rec = tmp_path / f"rec{ext}"
    rec.write_bytes(b"")
    assert ui.source_problem(rec) is None
    assert ui.source_problem(tmp_path) is None
