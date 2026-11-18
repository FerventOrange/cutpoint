"""Backend of ``cutpoint ui``: trust boundaries, review validation, media Range, job semantics."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("httpx")
import importlib.util
import tempfile

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from conftest import (
    link_dir,
    link_file,
    media_info,
    needs_ffmpeg,
    patch_ui,
    remap,
    set_config,
    write,
)
from cutpoint import (
    cache,
    capabilities,
    cli,
    config,
    edit,
    encode,
    errors,
    extras,
    ffmpeg_utils,
    folders,
    procs,
    runstate,
    ui,
)
from cutpoint.assistant import cliproxy, keys
from cutpoint.assistant import session as ai_session
from cutpoint.pano import candidates as pano_candidates
from cutpoint.pano import director as pano_director
from cutpoint.pano import project as pano_project
from cutpoint.pano import render as pano_render
from cutpoint.pano import stitch as pano_stitch
from cutpoint.ui import media as media_module
from cutpoint.ui.routes import assistant as assistant_routes
from cutpoint.utils import find_videos
from pano_helpers import x3clip
from proxy_helpers import FakeLauncher, not_running
from test_utils import MPEG_EXTS
from ui_helpers import JobSpy

PORT = 8765
# The longest wait for a real process to start, print, be cancelled or die: generous, as two
# interpreter starts can be slow under a loaded suite; a passing run never waits it out.
PROC_WAIT_S = 60.0


@pytest.fixture
def cpu_codecs(monkeypatch: pytest.MonkeyPatch) -> None:
    # job subprocesses inherit the CPU codecs: no NVENC/NVDEC contention under a loaded suite
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setenv("CUTPOINT_VIDEO_DECODER", "cpu")


@pytest.fixture
def client(cpu_codecs: None, ui_client: TestClient) -> TestClient:
    """The shared ``ui_client``, made after the codec variables are set."""
    return ui_client


@pytest.fixture(autouse=True)
def no_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxies encode with libx264 unless a test says otherwise: no GPU probe in tests."""
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("cpu"))


@pytest.fixture(autouse=True)
def no_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings reply reports where each provider key comes from: no keyring, so no D-Bus probe."""
    monkeypatch.setattr(
        keys, "_keyring", lambda: SimpleNamespace(get_password=lambda service, user: None)
    )


def make_project(client: TestClient, tmp_path: Path, files: dict[str, bytes]) -> tuple[str, Path]:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    res = client.post("/api/projects", json={"name": "My Trip", "source": str(folder)})
    assert res.status_code == 200, res.text
    return res.json()["id"], folder


X3_00 = "VID_20260101_100000_00_001.insv"
X3_10 = "VID_20260101_100000_10_001.insv"
X3_LRV = "LRV_20260101_100000_11_001.lrv"


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


X3_CLIPS = {X3_00: b"", X3_10: b"", X3_LRV: b""}


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


def test_create_project_refuses_ski_as_an_unknown_style(client: TestClient, tmp_path: Path) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "T", "source": str(folder), "style": "ski"})
    assert res.status_code == 400 and res.json()["detail"] == "unknown style 'ski'"
    assert client.get("/api/projects").json() == {"projects": []}


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


def wait_job(client: TestClient, timeout: float = 600) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        if jobs and jobs[-1]["status"] != "running":
            return jobs[-1]
        time.sleep(0.5)
    for job in client.get("/api/jobs").json()["jobs"]:
        if job["status"] == "running":
            client.post(f"/api/jobs/{job['id']}/cancel")  # do not leave it running past the test
    raise AssertionError("job did not finish")


def job_failure(job: dict) -> str:
    """What a job that did not complete said: its status, error and last log lines."""
    logs = "\n".join(job.get("logs", [])[-30:])
    return (
        f"job {job['status']}: {job.get('error')} ({job.get('error_code')})\n"
        f"steps: {job.get('progress')}\nlog:\n{logs}"
    )


def test_requests_must_be_loopback_and_carry_the_token(
    client: TestClient, dialog: FakeDialog
) -> None:
    """DNS rebinding / cross-site pages / tokenless callers get nothing."""
    assert client.get("/api/projects", headers={"Host": "evil.example:8765"}).status_code == 403
    assert client.get("/api/projects", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/session", headers={"Origin": "http://localhost:9"}).status_code == 403
    assert (
        client.get("/api/session", headers={"X-Cutpoint-Token": ""}).status_code == 403
    )  # no bootstrap
    assert client.get("/api/projects", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    bare = {"X-Cutpoint-Token": ""}
    assert client.post("/api/pick", params={"kind": "file"}, headers=bare).status_code == 403
    assert client.get("/api/settings", headers=bare).status_code == 403
    body = {"projects_dir": "x"}
    assert client.put("/api/settings", json=body, headers=bare).status_code == 403
    assert client.get("/api/settings", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    assert dialog.calls == []
    framed = client.get("/api/session")
    assert framed.headers["x-frame-options"] == "DENY"
    assert client.get("/api/projects").json() == {"projects": []}


def test_a_refused_request_says_whether_the_token_or_the_host_was_wrong(
    client: TestClient, tmp_path: Path
) -> None:
    """Cross-language contract: web/api.js TOKEN_DETAIL must equal the token detail below; it shows
    the "server restarted" banner and stops polling on it, and must not for the host detail
    (tests/js/api.test.mjs)."""
    token_detail = {"detail": "missing or wrong session token"}
    js_detail = re.search(r"\bTOKEN_DETAIL\s*=\s*(['\"])(.*?)\1", _web("api.js"))
    assert js_detail and js_detail[2] == token_detail["detail"]
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    media = client.get(f"/api/projects/{pid}").json()["sources"][0]["url"]
    wrong_t = media.split("?")[0] + "?t=stale"
    for wrong in ("", "stale"):
        bad = {"X-Cutpoint-Token": wrong}
        for url in ("/api/session", f"/api/projects/{pid}", wrong_t):
            res = client.get(url, headers=bad)
            assert (res.status_code, res.json()) == (403, token_detail), url
    host_detail = {"detail": "forbidden host or origin"}
    res = client.get("/api/session", headers={"Host": "evil.example:8765"})
    assert (res.status_code, res.json()) == (403, host_detail)
    res = client.get("/api/session", headers={"Origin": "http://evil.example"})
    assert (res.status_code, res.json()) == (403, host_detail)


def test_the_scrolling_columns_clip_their_visually_hidden_labels() -> None:
    """Regression (#354, #415): .vh labels are position: absolute. Unless .col is positioned, their
    containing block is the page, the column's overflow does not clip them, and a long moments
    list makes the page scroll past the Jobs bar on a wide window. Layout jsdom cannot see."""
    css = _web("app.css")
    assert _css_decls(css, ".vh")["position"] == "absolute"
    col = _css_decls(css, ".col")
    assert (col["position"], col["overflow"]) == ("relative", "auto")


def _theme_colours(css: str, selector: str) -> dict[str, str]:
    block = re.search(re.escape(selector) + r" \{(.*?)\n\}", css, re.S)
    assert block, selector
    return dict(re.findall(r"--([\w-]+): (#[0-9a-f]{6})", block[1]))


def _contrast(a: str, b: str) -> float:
    def lum(hex_colour: str) -> float:
        rgb = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_every_theme_keeps_its_text_and_controls_readable() -> None:
    """Contrast contract (#445, WCAG 1.4.3 and 1.4.11): in Graphite (the :root colours), Light and
    Blue, text is 4.5:1 and controls 3:1 on their grounds. The themes only override custom
    properties, so outside the theme blocks no rule holds a colour of its own (it would not
    follow the theme). Computed from app.css values: jsdom applies no stylesheet. Picking and
    storing a theme and the early script are in tests/js/graphite.test.mjs and workspace.test.mjs."""
    css = _web("app.css")
    graphite = _theme_colours(css, ":root")
    for name in ("graphite", "blue", "light"):
        t = (
            graphite
            if name == "graphite"
            else {**graphite, **_theme_colours(css, f':root[data-theme="{name}"]')}
        )
        text = [
            ("text", b) for b in ("slate", "panel", "raise", "deep", "picked", "hover", "log-bg")
        ]
        text += [
            (f, b) for f in ("muted", "amber-text", "bad", "ok") for b in ("slate", "panel", "deep")
        ]
        text += [("muted", "raise"), ("glow", "panel"), ("bad", "bad-bg"), ("amber-ink", "amber")]
        text += [
            ("amber-ink", "amber-hover"),
            ("danger-ink", "danger-btn"),
            ("danger-ink", "danger-btn-hover"),
        ]
        if name == "graphite":  # --faint is its dimmest text; the others alias it to --muted
            text += [("faint", b) for b in ("slate", "panel", "deep")]
        for fg, bg in text:
            assert _contrast(t[fg], t[bg]) >= 4.5, (name, fg, bg)
        # --amber-text: focus rings and amber text; --amber is the fill, with --amber-ink on it
        ui_parts = [("amber-text", b) for b in ("slate", "panel", "raise", "deep")]
        ui_parts += [
            ("glow", "deep"),
            ("danger", "deep"),
            ("muted", "well"),
            ("gold", "deep"),
            ("gold-edge", "deep"),
        ]
        if name != "blue":  # Blue keeps today's borders (below 3:1)
            ui_parts += [("line", "panel"), ("line", "slate"), ("line", "deep"), ("amber", "track")]
        for fg, bg in ui_parts:
            assert _contrast(t[fg], t[bg]) >= 3, (name, fg, bg)
    assert _css_decls(css, ':root[data-theme="light"]')["color-scheme"] == "light"
    # no colour outside the theme blocks (the backdrop keeps a fallback where it does not inherit)
    rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = re.sub(r":root[^{]*\{[^}]*\}", "", rules)
    rules = re.sub(r"var\(--backdrop, [^)]*\)\)", "", rules)
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b|rgba?\(\d", rules)
    for rule in re.findall(r"[^{}]*input\[type=range\][^{}]*\{[^}]*accent-color[^}]*\}", css):
        assert "accent-color: var(--amber-text)" in rule, rule


def test_the_page_files_are_revalidated_on_each_load(client: TestClient) -> None:
    """#415: with no Cache-Control the browser reused an app.css from before the #354 fix, judged
    fresh by its age, and the page scrolled past the Jobs bar again. Each load must revalidate.

    Layout check in a browser (no browser in the test suite): a project with 20+ moments and
    clips, the editor open, at 1915x950 and 1280x720 the document's scrollHeight equals
    innerHeight and #jobs ends at the bottom edge; at 960px wide the page scrolls as before."""
    for path in ("/", "/index.html", "/app.css", "/app.js", "/api.js"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["cache-control"] == "no-cache", path
        again = client.get(path, headers={"If-None-Match": res.headers["etag"]})
        assert again.status_code == 304, path
    assert client.get("/api/session").headers["cache-control"] == "no-store"


def test_project_ids_cannot_leave_the_projects_folder(client: TestClient, tmp_path: Path) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    for bad_id in ("..", "%2e%2e", "a%2Fb", pid + "%00"):
        assert client.get(f"/api/projects/{bad_id}").status_code in (404, 422)
    assert client.get(f"/api/projects/{pid}").status_code == 200


def test_review_is_validated_and_saved_atomically(client: TestClient, tmp_path: Path) -> None:
    """Only project sources, real ranges inside the source; bad input leaves the saved review."""
    private = tmp_path / "private.mp4"
    private.write_bytes(b"secret")
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    src = str((folder / "a.mp4").resolve())
    p = ui.project_dir(pid)
    # highlights.json names a file outside the picked folder: it must not become a source.
    cache.save_json(
        p / "highlights.json",
        {
            "settings": {"inputs": [str(private)]},
            "sources": [
                {"path": src, "duration": 10.0},
                {"path": str(private), "duration": 10.0},
                {"path": "/etc/hostname", "duration": 1.0},
            ],
            "candidates": [],
        },
    )
    assert [s["path"] for s in client.get(f"/api/projects/{pid}").json()["sources"]] == [src]

    def put(*segments: dict):
        return client.put(f"/api/projects/{pid}/review", json={"segments": list(segments)})

    ok = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    assert put(ok).status_code == 200
    saved = client.get(f"/api/projects/{pid}").json()["review"]["segments"]
    assert saved == [{**ok, "start": 1.0, "end": 4.5}]
    # #324: the editor plays the whole source, so a trim may reach far past the analyzed moment.
    assert put({**ok, "end": 9.9}).status_code == 200
    assert put(ok).status_code == 200
    for broken in (
        {**ok, "start": 5, "end": 5},  # empty
        {**ok, "end": 10.5},  # past the source
        {**ok, "source": str(private)},  # not the project's
        {**ok, "start": "1"},  # not a number
    ):
        assert put(broken).status_code == 400
    assert put(ok, ok).status_code == 400  # duplicate id
    assert client.get(f"/api/projects/{pid}").json()["review"]["segments"] == saved


def test_the_target_length_is_saved_with_the_project_and_survives_save(
    client: TestClient, tmp_path: Path
) -> None:
    """The sequence's target length lives in ui/review.json: validated, cleared by null and kept
    by a Save of the sequence."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    src = str((folder / "a.mp4").resolve())
    p = ui.project_dir(pid)
    cache.save_json(p / "highlights.json", {"sources": [{"path": src, "duration": 10.0}]})
    url = f"/api/projects/{pid}/target"

    def put(seconds):
        return client.put(url, json={"seconds": seconds})

    def stored():
        return (client.get(f"/api/projects/{pid}").json()["review"] or {}).get("target")

    assert put(240).json() == {"target": 240.0}
    assert stored() == 240.0
    seg = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    assert client.put(f"/api/projects/{pid}/review", json={"segments": [seg]}).status_code == 200
    assert stored() == 240.0  # Save keeps it
    review = client.get(f"/api/projects/{pid}").json()["review"]
    assert [s["id"] for s in review["segments"]] == ["h001"]
    assert put(36000).status_code == 200
    assert put(0.5).status_code == 200
    for bad in (0, -1, 36000.5, 10**400, "240", True, [], {}):
        assert put(bad).status_code == 400, bad
        assert stored() == 0.5
    for raw in ("NaN", "Infinity", "-Infinity"):  # not JSON, but Python's parser takes them
        res = client.put(
            url, content=f'{{"seconds": {raw}}}', headers={"content-type": "application/json"}
        )
        assert res.status_code in (400, 422), raw
        assert stored() == 0.5
    assert client.put(url, json={}).status_code == 400
    assert put(None).json() == {"target": None}
    assert stored() is None
    assert client.get(f"/api/projects/{pid}").json()["review"]["segments"][0]["id"] == "h001"


def test_save_keeps_the_target_when_review_json_is_locked_for_a_moment(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short lock on review.json (first read raises PermissionError) must not drop the target."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    src = str((folder / "a.mp4").resolve())
    p = ui.project_dir(pid)
    cache.save_json(p / "highlights.json", {"sources": [{"path": src, "duration": 10.0}]})
    assert client.put(f"/api/projects/{pid}/target", json={"seconds": 240}).status_code == 200
    file = p / "ui" / "review.json"
    real = Path.read_text
    calls: list[int] = []

    def flaky(self: Path, *a, **kw) -> str:
        if self == file and not calls:
            calls.append(1)
            raise PermissionError("locked")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", flaky)
    seg = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    assert client.put(f"/api/projects/{pid}/review", json={"segments": [seg]}).status_code == 200
    assert calls == [1]
    assert json.loads(file.read_text(encoding="utf-8"))["target"] == 240.0


def test_the_target_is_not_saved_over_a_review_file_that_cannot_be_read(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt, non-object or unreadable review.json gives 409 and is never rewritten."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    file = ui.project_dir(pid) / "ui" / "review.json"
    file.parent.mkdir(exist_ok=True)
    url = f"/api/projects/{pid}/target"
    for raw in (b'{"segments": [{"id": "h00', b"[1, 2]", b"\xff\xfe\x00bad"):
        file.write_bytes(raw)
        for seconds in (60, None):
            assert client.put(url, json={"seconds": seconds}).status_code == 409, raw
        assert file.read_bytes() == raw

    good = b'{"segments": []}'
    file.write_bytes(good)

    def locked(path, hint=""):
        raise PermissionError("locked")

    monkeypatch.setattr(cache, "read_json", locked)
    assert client.put(url, json={"seconds": 60}).status_code == 409
    assert file.read_bytes() == good


def test_moment_ids_survive_reanalysis_without_relinking_saved_trims(
    client: TestClient, tmp_path: Path
) -> None:
    """Rank ids (h001) are reassigned by analysis; a saved trim must never follow one."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x", "b.mp4": b"x"})
    a, b = (str((folder / n).resolve()) for n in ("a.mp4", "b.mp4"))
    p = ui.project_dir(pid)

    def analyse(*cands: tuple[str, str, float, float]) -> dict[tuple[str, float, float], str]:
        cache.save_json(
            p / "highlights.json",
            {
                "sources": [{"path": a, "duration": 400.0}, {"path": b, "duration": 400.0}],
                "candidates": [
                    {"id": i, "source": s, "start": t0, "end": t1} for i, s, t0, t1 in cands
                ],
            },
        )
        got = client.get(f"/api/projects/{pid}").json()["highlights"]["candidates"]
        return {(c["source"], c["start"], c["end"]): c["id"] for c in got}

    first = analyse(("h001", a, 10, 20), ("h002", a, 30, 40))
    old = {"id": "h001", "source": a, "start": 11.0, "end": 19.0, "label": "old"}
    assert client.put(f"/api/projects/{pid}/review", json={"segments": [old]}).status_code == 200

    again = analyse(("h001", b, 300.0, 310.0), ("h002", a, 30.0, 40.0), ("h003", a, 10.0, 20.0))
    assert "h001" not in again.values()
    assert again[(a, 10, 20)] == first[(a, 10, 20)]  # reranked, same moment
    assert first[(a, 10, 20)] != first[(a, 30, 40)]
    assert again[(b, 300, 310)] not in first.values()
    assert client.get(f"/api/projects/{pid}").json()["review"]["segments"] == [old]


def test_moment_cards_get_the_cli_candidate_id_that_ids_takes(tmp_path: Path) -> None:
    """#487: the payload keeps the CLI id (h001) next to the internal id, and Candidate ids
    (``clips --ids``) takes exactly what the cards show."""
    src = str(tmp_path / "a.mp4")
    hl = {
        "candidates": [{"id": f"h00{i}", "source": src, "start": i, "end": i + 5.0} for i in (1, 2)]
    }
    view = ui.moments_view(hl)["candidates"]
    assert [c["cli_id"] for c in view] == ["h001", "h002"]
    assert all(c["id"] not in ("h001", "h002") for c in view)  # the internal id stays the hash
    shown = [c["cli_id"] for c in view][::-1]
    assert ui._auto_picks({"ids": shown}, hl) == ["--ids", "h002", "h001"]


def test_moments_view_looks_up_each_source_once_and_keeps_the_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A742: a source's real path is looked up once per view, not once per moment (with
    thousands of moments it was most of the project reply's time); the ids stay moment_id's."""
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    spans = [(a, 1.0, 5.0), (b, 2.0, 6.0), (a, 10.0, 15.0), (a, 20.0, 25.5), (b, 30.0, 31.0)]
    hl = {"candidates": [{"source": s, "start": t0, "end": t1} for s, t0, t1 in spans]}
    want = [ui.moment_id(s, t0, t1) for s, t0, t1 in spans]
    asked: list[str] = []
    realpath = os.path.realpath

    def counted(path: str) -> str:
        asked.append(path)
        return realpath(path)

    monkeypatch.setattr(os.path, "realpath", counted)
    view = ui.moments_view(hl)["candidates"]
    monkeypatch.undo()
    assert sorted(asked) == [a, b]
    assert [c["id"] for c in view] == want


@pytest.mark.parametrize(
    ("argv", "flag", "expected"),
    [
        (["analyze", "a.mp4", "--style", "funny", "--json"], "--style", ["funny"]),
        (["analyze", "a.mp4", "--skip", "crashes", "dialogue"], "--skip", ["crashes", "dialogue"]),
        (["analyze", "a.mp4", "--skip=crashes", "dialogue"], "--skip", ["crashes"]),
        (["clips", "p", "--top=8"], "--top-n", []),  # another flag's prefix is not it
        (["clips", "p", "--top8"], "--top", []),
        # repeated: argparse keeps the last, in either form
        (["analyze", "a.mp4", "--style", "gaming", "--style=funny"], "--style", ["funny"]),
        (["analyze", "a.mp4", "--style=gaming", "--style", "funny", "-v"], "--style", ["funny"]),
    ],
)
def test_a_recorded_command_line_gives_its_flag_values_in_both_forms(
    argv: list[str], flag: str, expected: list[str]
) -> None:
    """A269: ``--flag=value`` is read as argparse reads it."""
    from cutpoint.ui.files import _values  # noqa: PLC0415

    assert _values(argv, flag) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3.0),
        (2.5, 2.5),
        (True, None),
        (None, None),
        (math.nan, None),
        (10**400, None),  # too large for a float
    ],
)
def test_finite_number_takes_only_finite_ints_and_floats(value: Any, expected: Any) -> None:
    from cutpoint.ui.common import finite_number  # noqa: PLC0415

    got = finite_number(value)
    assert got == expected and (got is None or type(got) is float)


def test_a_flag_or_target_check_refuses_nan_infinity_and_bools() -> None:
    """A415: every UI number check reads a value as :func:`finite_number` does."""
    from cutpoint.ui.analysis import _color_state, _seconds  # noqa: PLC0415
    from cutpoint.ui.pano_previews import _pano_reasons  # noqa: PLC0415
    from cutpoint.ui.projects import _secs, _valid_target  # noqa: PLC0415

    for v in (math.nan, math.inf, 10**400, True, "5"):
        assert _secs(v) is None and not _valid_target(v)
        assert _color_state({"mode": "auto", "strength": v}) is None
        with pytest.raises(HTTPException):
            _seconds({"max_total": v}, "max_total", 60.0)
    assert _secs(5) == 5.0 and _valid_target(60) and _seconds({"x": 2}, "x", 60.0) == 2.0
    assert _color_state({"mode": "auto", "strength": 1}) == {"mode": "auto", "strength": 1.0}
    # a NaN share is left out of the flag's reason instead of breaking the listing
    reasons = _pano_reasons(["sky"], {"sky": "mostly sky"}, {"sky": math.nan})
    assert reasons == [{"flag": "sky", "text": "mostly sky"}]


@pytest.mark.parametrize(
    ("device", "expected"), [("cuda", "medium"), ("cpu", "small"), (None, "small")]
)
def test_the_ui_default_whisper_model_is_transcribes(device: Any, expected: str) -> None:
    """A417: the UI asks :func:`cutpoint.transcribe.default_model` for the device the analysis
    ran on (an older analysis without one: the CPU's)."""
    from cutpoint.ui.analysis import _default_model  # noqa: PLC0415

    assert _default_model({"whisper_device": device}) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "assistant"), ("", "assistant"), ("assistant", "assistant"), ("vision", "vision")],
)
def test_a_role_is_assistant_or_vision(value: Any, expected: str) -> None:
    from cutpoint.ui.routes.settings import role_of  # noqa: PLC0415

    assert role_of(value) == expected


@pytest.mark.parametrize("value", ["chat", 1, ["vision"]])
def test_another_role_is_refused(value: Any) -> None:
    from cutpoint.ui.routes.settings import role_of  # noqa: PLC0415

    with pytest.raises(HTTPException) as err:
        role_of(value)
    assert err.value.status_code == 400 and err.value.detail == "role must be assistant or vision"


def test_plain_json_writes_the_bytes_fastapis_own_reply_writes() -> None:
    """A742: the project reply skips FastAPI's value-by-value copy (``jsonable_encoder``) but
    writes the same bytes, a value JSON has no type for included; NaN is refused as before."""
    content = {
        "a": [1, 2.5, None, True, "é ✓ \u2028"],
        "t": (1, "x"),
        "p": Path("x") / "y.mp4",
        "n": {"k": {"deep": [{}, []]}},
        3: "an int key",
    }
    assert ui.PlainJSON(content).body == JSONResponse(jsonable_encoder(content)).body
    with pytest.raises(ValueError):
        ui.PlainJSON({"x": float("nan")})
    with pytest.raises(ValueError):
        JSONResponse(jsonable_encoder({"x": float("nan")}))


def test_project_reply_follows_highlights_json_rewritten_with_the_same_size_and_time(
    client: TestClient, tmp_path: Path
) -> None:
    """A742: the project reply reads highlights.json each time, so a rewrite shows at once even
    when the file keeps its size and modification time (a coarse clock, a copied file)."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    src = str((folder / "a.mp4").resolve())
    hl = p / "highlights.json"

    def analyse(label: str) -> None:
        moment = {"id": "h001", "source": src, "start": 1.0, "end": 5.0, "label": label}
        cache.save_json(hl, {"candidates": [moment]})

    analyse("aaaa")
    st = hl.stat()
    first = client.get(f"/api/projects/{pid}")
    assert first.headers["content-type"] == "application/json"
    analyse("bbbb")
    os.utime(hl, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (hl.stat().st_size, hl.stat().st_mtime_ns) == (st.st_size, st.st_mtime_ns)
    second = client.get(f"/api/projects/{pid}").json()
    assert [c["label"] for c in first.json()["highlights"]["candidates"]] == ["aaaa"]
    assert [c["label"] for c in second["highlights"]["candidates"]] == ["bbbb"]
    assert second["ranking_key"] != first.json()["ranking_key"]


def test_media_supports_range_and_only_serves_project_sources(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = media_info(duration=10.0)  # h264 yuv420p + one AAC track: the browser plays it
    patch_ui(monkeypatch, "probe", lambda _path: info)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"0123456789"})
    source = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert source["status"] == "unknown"  # opening a project probes nothing
    client.post(f"/api/projects/{pid}/sources/{source['id']}/prepare")
    source = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert source["status"] == "ready"
    res = client.get(source["url"], headers={"Range": "bytes=2-5"})
    assert (res.status_code, res.content) == (206, b"2345")
    assert res.headers["content-range"] == "bytes 2-5/10"
    assert client.get(source["url"], headers={"Range": "bytes=50-"}).status_code == 416
    bare = source["url"].split("?")[0]  # no ?t= token, and the header token is blanked
    assert client.get(bare, headers={"X-Cutpoint-Token": ""}).status_code == 403
    unknown = source["url"].replace(source["url"].split("/")[-1].split("?")[0], "0" * 12)
    assert client.get(unknown).status_code == 404


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.mark.slow
@needs_ffmpeg
def test_proxy_is_built_only_for_the_chosen_source_and_keeps_every_audio_track(
    client: TestClient, tmp_path: Path
) -> None:
    """Opening a project builds nothing; preparing one source mixes all its audio tracks."""
    multi, other = tmp_path / "a.mp4", tmp_path / "b.mkv"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=d=3:s=160x120:r=25"),
        *("-f", "lavfi", "-i", "sine=f=440:d=3", "-f", "lavfi", "-i", "sine=f=880:d=3"),
        *("-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p"),
        *("-c:a", "aac", str(multi)),
    )  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=160x120:r=25", "-c:v", "mpeg4", str(other))
    pid, _ = make_project(
        client, tmp_path, {"a.mp4": multi.read_bytes(), "b.mkv": other.read_bytes()}
    )

    def sources() -> dict[str, dict]:
        return {s["name"]: s for s in client.get(f"/api/projects/{pid}").json()["sources"]}

    opened = sources()
    assert (opened["a.mp4"]["status"], opened["b.mkv"]["status"]) == ("unknown",) * 2
    assert not (ui.project_dir(pid) / ui.UI_DIR / "proxy").exists()  # nothing was queued
    assert client.get(opened["a.mp4"]["url"]).status_code == 409

    client.post(f"/api/projects/{pid}/sources/{opened['a.mp4']['id']}/prepare")
    # The proxy encodes at low priority (below normal on Windows, nice 10 elsewhere): on a loaded
    # machine its 3 s encode can wait minutes for the CPU, so the deadline is wait_job's.
    deadline = time.monotonic() + 600
    while (status := sources()["a.mp4"]["status"]) == "preparing":
        assert time.monotonic() < deadline, "the proxy was still preparing after 600 s"
        time.sleep(0.3)
    after = sources()
    assert (after["a.mp4"]["status"], after["b.mkv"]["status"]) == ("ready", "unknown"), status

    url = after["a.mp4"]["url"]
    assert client.get(url, headers={"Range": "bytes=0-99"}).status_code == 206
    played = tmp_path / "played.mp4"
    played.write_bytes(client.get(url).content)
    # #324's whole-source scrubbing relies on the proxy covering the full source, not a moment.
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(played)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert float(duration) == pytest.approx(3.0, abs=0.15)
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(played),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    spectrum = np.abs(np.fft.rfft(np.frombuffer(pcm, dtype=np.int16).astype(float)))
    freqs = np.fft.rfftfreq(len(pcm) // 2, 1 / 8000)
    for tone in (440, 880):
        assert spectrum[abs(freqs - tone) < 5].max() > 20 * np.median(spectrum), tone


@pytest.mark.slow
@needs_ffmpeg
def test_reanalyze_keeps_recorded_excludes_and_recursion(
    client: TestClient, tmp_path: Path
) -> None:
    """A command-line project re-analyzed from the UI keeps its subfolders and its exclusions."""
    clip = tmp_path / "clip.mp4"  # silent, with scene cuts: nothing to transcribe or download
    ffmpeg(
        *("-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=2"),
        *("-f", "lavfi", "-i", "color=c=black:s=160x120:r=25:d=2"),
        *("-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=2"),
        *("-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-c:v", "libx264"),
        *("-pix_fmt", "yuv420p", str(clip)),
    )  # fmt: skip
    footage = tmp_path / "footage"
    (footage / "sub").mkdir(parents=True)
    for rel in ("keep.mp4", "DVR_skip.mp4", "sub/nested.mp4", "sub/DVR_nested.mp4"):
        shutil.copy(clip, footage / rel)
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    settings = {"inputs": [str(footage.resolve())], "exclude": ["*DVR*"], "recursive": True}
    # Not a game or action style: those run HUD OCR with region discovery (about 80 s a source)
    # or CLIP keyframes, which this source-set check does not need.
    style = "commentary"
    cache.save_json(p / "highlights.json", {"style": style, "settings": settings, "sources": []})

    assert client.post("/api/projects/cli-made/jobs", json={"action": "analyze"}).status_code == 200
    job = wait_job(client)
    assert job["status"] == "completed", job_failure(job)
    names = {s["name"] for s in client.get("/api/projects/cli-made").json()["sources"]}
    assert names == {"keep.mp4", "nested.mp4"}


@pytest.mark.parametrize(("recover", "passed"), [(True, True), (False, False), ("yes", False)])
def test_reanalyze_keeps_a_recorded_recover(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recover: object,
    passed: bool,
) -> None:
    """Analyze again on a project made with --recover keeps the parts past a read error."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "funny"
    hl["settings"]["recover"] = recover
    cache.save_json(p / "highlights.json", hl)
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert ("--recover" in captured[0]) is passed
    cli.build_parser().parse_args(captured[0][3:])


ANALYZE_DEFAULTS = {
    "model": None,
    "language": None,
    "max_candidates": None,
    "no_scenes": False,
    "skip": None,
    "must_include_crash": None,
    "recover": False,
}
RECORDED = {  # an analysis run with every Analyze option but --skip (not recorded)
    "whisper_model": "large-v3",
    "whisper_device": "cpu",
    "language": "de",
    "max_candidates": 30,
    "scenes": False,
    "recover": True,
}


@pytest.mark.parametrize(
    ("settings", "body", "parsed"),
    [
        (
            {},
            {
                "model": " turbo ",
                "language": "en",
                "max_candidates": 12,
                "no_scenes": True,
                "skip": ["crashes", "reactions"],
                "must_include_crash": 0.25,
                "recover": True,
            },
            {
                "model": "turbo",
                "language": "en",
                "max_candidates": 12,
                "no_scenes": True,
                "skip": ["reactions", "crashes"],
                "must_include_crash": 0.25,
                "recover": True,
            },
        ),
        # Analyze again from a request without them: what the analysis recorded
        (
            RECORDED,
            {},
            {
                "model": "large-v3",
                "language": "de",
                "max_candidates": 30,
                "no_scenes": True,
                "recover": True,
            },
        ),
        # the page cleared them: the CLI defaults (a recorded --recover is dropped too)
        (
            RECORDED,
            {
                "model": "",
                "language": None,
                "max_candidates": None,
                "no_scenes": False,
                "recover": False,
                "skip": [],
            },
            {},
        ),
        # each refused, nothing started
        ({}, [{"skip": ["vision"]}, {"model": "../x"}], None),
    ],
    ids=["set", "recorded", "cleared", "invalid"],
)
def test_analyze_options_reach_the_analyze_command_line_unchanged(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: dict,
    body: dict | list[dict],
    parsed: dict | None,
) -> None:
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "funny"
    hl["settings"].update(settings)
    cache.save_json(p / "highlights.json", hl)
    if settings:  # the page shows what the analysis used
        assert client.get("/api/projects/cli").json()["analyze_options"] == {
            **ANALYZE_DEFAULTS, "model": "large-v3", "language": "de", "max_candidates": 30,
            "no_scenes": True, "recover": True, "skip": [], "vision_detect_all": False,
        }  # fmt: skip
    captured = JobSpy(monkeypatch, check=False).args
    if isinstance(body, list):
        for bad_body, message in zip(
            body, ("skip must be a list of", "model must be a whisper"), strict=True
        ):
            res = client.post("/api/projects/cli/jobs", json={"action": "analyze", **bad_body})
            assert res.status_code == 400 and message in res.json()["detail"]
        assert not captured
        return
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200, res.text
    a = cli.build_parser().parse_args(_cli_args(captured[0]))
    assert {k: getattr(a, k) for k in ANALYZE_DEFAULTS} == {**ANALYZE_DEFAULTS, **parsed}


def _auto_project(tmp_path: Path, **settings: object) -> Path:
    """A command-line project analyzed with ``--style auto`` (it chose gaming)."""
    det = {"style": "gaming", "category": "gameplay"}
    settings = {"style_requested": "auto", **settings}
    return cli_project(tmp_path, settings=settings, style="gaming", detection=det)[0]


@pytest.mark.parametrize(
    ("settings", "config", "want"),
    [
        ({"vision_detect_all": True}, {}, True),  # recorded by the analysis
        ({}, {"vision_detect_all": True}, False),  # an auto analysis without it: off
    ],
)
def test_analyze_options_report_vision_detect_all(
    client: TestClient, tmp_path: Path, settings: dict, config: dict, want: bool
) -> None:
    _auto_project(tmp_path, **settings)
    set_config(ui.paths.home(), **config)
    opts = client.get("/api/projects/cli").json()["analyze_options"]
    assert opts["vision_detect_all"] is want


def test_analyze_options_vision_detect_all_defaults_to_the_config_before_an_auto_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    cli_project(tmp_path)  # no analysis style recorded
    assert client.get("/api/projects/cli").json()["analyze_options"]["vision_detect_all"] is False
    set_config(ui.paths.home(), vision_detect_all=True)
    assert client.get("/api/projects/cli").json()["analyze_options"]["vision_detect_all"] is True
    set_config(ui.paths.home(), vision_detect_all="yes")  # invalid: the helper does not raise
    assert ui._vision_detect_all(None) is False


@pytest.mark.parametrize(
    ("recorded", "body", "flag"),
    [
        (False, {"vision_detect_all": True}, "--vision-detect-all"),
        (True, {"vision_detect_all": False}, "--no-vision-detect-all"),
        (True, {}, "--vision-detect-all"),  # Analyze again repeats the analysis's
        (False, {}, "--no-vision-detect-all"),  # explicit, even off: config.json may differ
        (False, {"vision_detect_all": None}, None),  # the page sent none: the CLI's default
    ],
)
def test_an_auto_analyze_job_sends_vision_detect_all_explicitly(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recorded: bool,
    body: dict,
    flag: str | None,
) -> None:
    _auto_project(tmp_path, **({"vision_detect_all": True} if recorded else {}))
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert _after(args, "--style") == "auto"
    got = [a for a in args if a in ("--vision-detect-all", "--no-vision-detect-all")]
    assert got == ([flag] if flag else [])


@pytest.mark.parametrize("style", ["funny", "gaming"])
def test_other_styles_send_no_vision_detect_all(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, style: str
) -> None:
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = style
    hl["settings"]["vision_detect_all"] = True
    cache.save_json(p / "highlights.json", hl)
    captured = JobSpy(monkeypatch, check=False).args
    body = {"action": "analyze", "vision_detect_all": True}
    assert client.post("/api/projects/cli/jobs", json=body).status_code == 200
    assert not [a for a in captured[0] if a.endswith("-vision-detect-all")]


def test_a_non_boolean_vision_detect_all_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _auto_project(tmp_path)
    captured = JobSpy(monkeypatch, check=False).args
    for bad_value in ("yes", 1, [True]):
        body = {"action": "analyze", "vision_detect_all": bad_value}
        res = client.post("/api/projects/cli/jobs", json=body)
        assert res.status_code == 400
        assert "vision_detect_all must be true or false" in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(("recorded", "flag"), [(True, "--vision-detect-all"), (False, None)])
def test_one_click_edit_repeats_an_auto_analyses_vision_detect_all(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recorded: bool,
    flag: str | None,
) -> None:
    _auto_project(tmp_path, **({"vision_detect_all": True} if recorded else {}))
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", "kind": "clips"})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert args[0] == "edit"
    got = [a for a in args if a.endswith("-vision-detect-all")]
    assert got == [flag or "--no-vision-detect-all"]


def test_the_analyze_skip_is_kept_for_analyze_again_and_one_click_edit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """highlights.json does not record --skip: the UI keeps it for that analysis only, so
    edit (which analyzes again) leaves the same steps out, and a command-line re-run drops it."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl.update(style="funny", created="2026-10-01T10:00:00")
    cache.save_json(p / "highlights.json", hl)
    spy = JobSpy(monkeypatch)
    captured, done = spy.args, spy.on_success
    body = {"action": "analyze", "skip": ["crashes", "keyframes"]}
    assert client.post("/api/projects/cli/jobs", json=body).status_code == 200
    assert _cli_args(captured[0])[-3:] == ["--skip", "keyframes", "crashes"]
    hl["created"] = "2026-10-01T11:00:00"  # what that analysis wrote
    cache.save_json(p / "highlights.json", hl)
    done[0]()
    view = client.get("/api/projects/cli").json()
    assert view["analyze_options"]["skip"] == ["keyframes", "crashes"]
    assert view["preset_problem"] is None  # edit takes --skip
    assert (
        client.post("/api/projects/cli/jobs", json={"action": "edit", "kind": "clips"}).status_code
        == 200
    )
    assert _cli_args(captured[1])[-3:] == ["--skip", "keyframes", "crashes"]

    hl["created"] = "2026-10-02T09:00:00"  # analyzed again on the command line
    cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["analyze_options"]["skip"] == []
    assert client.post("/api/projects/cli/jobs", json={"action": "analyze"}).status_code == 200
    assert "--skip" not in captured[2]


def test_a_ski_analysis_without_no_scenes_keeps_presets_and_scene_detection_on(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: ski never detects scenes, so its analysis records "scenes": false without
    --no-scenes; that turned the presets off with the --no-scenes reason."""
    meta = {"style": "action-cam", "source": str(tmp_path / "clip.mp4")}
    settings = {"scenes": False, "must_include_crash": 0.6}  # the action-cam style's own
    res, captured, _, _ = preset_job(
        client,
        monkeypatch,
        tmp_path,
        {"kind": "clips"},
        settings=settings,
        meta=meta,
        style="action-cam",
    )
    assert res.status_code == 200, res.text
    _cli_args(captured[0])
    view = client.get("/api/projects/cli").json()
    assert view["preset_problem"] is None and view["must_include_crash_default"] == 0.6
    opts = view["analyze_options"]
    assert (opts["no_scenes"], opts["must_include_crash"]) == (False, None)


def test_an_auto_project_detected_as_action_cam_keeps_presets_and_scene_detection_on(
    client: TestClient, tmp_path: Path
) -> None:
    """An Auto project's detected style (not "auto") decides whether "scenes": false means
    --no-scenes."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "action-cam"
    hl["settings"]["scenes"] = False
    cache.save_json(p / "highlights.json", hl)
    write(p / "ui" / "meta.json", json.dumps({"style": "auto", "source": str(tmp_path / "c.mp4")}))
    view = client.get("/api/projects/cli").json()
    assert view["preset_problem"] is None
    assert view["analyze_options"]["no_scenes"] is False


@pytest.mark.parametrize(
    ("saved", "flags"),
    [
        (
            {"vision_top": 5, "ollama_url": "http://h:1", "ollama_model": "m"},
            ["--vision-top=5", "--ollama-url=http://h:1", "--ollama-model=m"],
        ),
        ({"vision_top": True, "ollama_url": "", "ollama_model": ""}, []),
        ({"vision_top": 0}, []),
    ],
    ids=["kept", "invalid-true-and-empty", "zero"],
)
def test_reanalyze_passes_recorded_vision_and_ollama_options(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved: dict,
    flags: list[str],
) -> None:
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    footage = tmp_path / "clip.mp4"
    footage.write_bytes(b"x")
    settings = {"inputs": [str(footage)], **saved}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings, "sources": []})
    captured = JobSpy(monkeypatch, check=False).args
    assert client.post("/api/projects/cli-made/jobs", json={"action": "analyze"}).status_code == 200
    args = captured[0]
    assert [a for a in args if a.startswith(("--vision-top", "--ollama"))] == flags


@pytest.mark.parametrize(
    ("body", "saved", "cfg", "vision"),
    [
        ({}, {}, None, None),  # no key in config.json: no flag, the CLI asks unsure sources
        ({}, {}, False, False),
        ({}, {}, True, True),  # Analyze again: the Settings value
        ({}, {"vision_requested": True}, None, None),  # not what the last run asked
        ({}, {"vision_requested": False}, True, True),
        ({"vision": True}, {}, False, False),  # no per-run choice: a body flag is ignored
        ({"vision": False}, {}, True, True),
    ],
)
def test_analyze_job_passes_the_settings_vision_explicitly(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    saved: dict,
    cfg: bool | None,
    vision: bool | None,
) -> None:
    """--vision or --no-vision as Settings says when the job starts; neither when config.json
    has no ``vision`` key."""
    if cfg is not None:
        set_config(home, vision=cfg)
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    footage = tmp_path / "clip.mp4"
    footage.write_bytes(b"x")
    settings = {"inputs": [str(footage)], **saved}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings, "sources": []})
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli-made/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200
    assert ("--vision" in captured[0]) is (vision is True)
    assert ("--no-vision" in captured[0]) is (vision is False)


def test_put_settings_saves_visual_index_and_jobs_pass_it_explicitly(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """visual_index (#798): saved by Settings, shown by GET, and --visual-index or
    --no-visual-index on every analysis and preset the page starts."""
    assert client.get("/api/settings").json()["visual_index"] is False
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    captured = JobSpy(monkeypatch, check=False).args
    for value in (True, False):
        res = client.put("/api/settings", json={"visual_index": value})
        assert res.status_code == 200, res.text
        assert res.json()["visual_index"] is value
        assert client.get("/api/settings").json()["visual_index"] is value
        assert config.visual_index_default() is value  # the CLI default too
        assert (
            client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"}).status_code == 200
        )
        assert ("--visual-index" in captured[-1]) is value
        assert ("--no-visual-index" in captured[-1]) is not value
    for bad_value in ("yes", 1, None):
        assert client.put("/api/settings", json={"visual_index": bad_value}).status_code == 400


def test_analyze_follows_settings_saved_just_before(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saving Settings changes the next analysis at once, without a reload or restart."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    captured = JobSpy(monkeypatch, check=False).args
    for value in (True, False, True):
        assert client.put("/api/settings", json={"vision": value}).status_code == 200
        res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
        assert res.status_code == 200
    assert [("--vision" in a, "--no-vision" in a) for a in captured] == [
        (True, False),
        (False, False),  # unticked: the key is gone, so neither flag
        (True, False),
    ]


@pytest.mark.slow
def test_one_job_at_a_time_and_cancel_kills_the_process_group(tmp_path: Path) -> None:
    script = (
        "import subprocess, sys, time\n"
        # longer than PROC_WAIT_S: a child left alive cannot pass the death check by exiting
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(600)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", script])
    child = 0
    try:
        # two interpreter starts can be slow under a loaded suite
        wait_until(lambda: job.logs or job.status != "running", PROC_WAIT_S, "the child's pid line")
        assert job.logs, (job.status, job.error)
        child = int(job.logs[0].split()[1])
        with pytest.raises(HTTPException) as busy:
            jobs.start("p", "clips", [sys.executable, "-c", "pass"])
        assert busy.value.status_code == 409
        cancelled = jobs.cancel(job.id)
        # cancel returns at once; the killer thread and the watcher finish in the background
        wait_until(lambda: cancelled.status == "cancelled", PROC_WAIT_S, "the job to be cancelled")
        assert job.proc is not None and job.proc.poll() is not None
        if sys.platform.startswith("linux"):  # a reaped-or-zombie child counts as dead
            wait_until(lambda: _linux_dead(child), PROC_WAIT_S, f"the child {child} to die")
    finally:  # what a failed cancel left running
        if child and sys.platform.startswith("linux") and not _linux_dead(child):
            os.kill(child, signal.SIGKILL)
        elif child and os.name == "nt" and _pid_alive(child):
            subprocess.run(["taskkill", "/PID", str(child), "/T", "/F"], capture_output=True)
        if job.proc is not None and job.proc.poll() is None:
            procs.kill_tree(job.proc, 0.0)
    failing = jobs.start(
        "p", "analyze", [sys.executable, "-c", "print('12:00:00 ERROR boom')\nraise SystemExit(1)"]
    )
    assert failing.done.wait(20)
    assert (failing.status, failing.error) == ("failed", "boom")


@pytest.mark.parametrize(
    ("event_sent", "exits_in_time", "job", "expected"),
    [
        (1, False, 777, ["ctrl_break", "wait", "terminate", "taskkill"]),  # job object first
        (1, False, None, ["ctrl_break", "wait", "taskkill"]),  # ignores the event
        (1, True, None, ["ctrl_break", "wait", "taskkill"]),  # taskkill still follows
        (0, False, None, ["ctrl_break", "taskkill"]),  # no console: nothing to wait for
    ],
    ids=["job-object", "ignores-the-event", "soft-stop-works", "no-console"],
)
def test_windows_cancel_tries_ctrl_break_then_the_job_object_then_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    event_sent: int,
    exits_in_time: bool,
    job: int | None,
    expected: list[str],
) -> None:
    calls: list[tuple[str, object]] = []

    def wait(timeout: float) -> int:
        calls.append(("wait", timeout))
        if exits_in_time:
            return 130
        raise subprocess.TimeoutExpired("job", timeout)

    kernel32 = SimpleNamespace(
        GenerateConsoleCtrlEvent=lambda event, pid: (
            calls.append(("ctrl_break", (event, pid))) or event_sent
        ),
        TerminateJobObject=lambda handle, code: calls.append(("terminate", (handle, code))) or 1,
    )
    fake_ctypes = SimpleNamespace(WinDLL=lambda name: kernel32, c_void_p=lambda value: value)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(procs, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(procs, "signal", SimpleNamespace(CTRL_BREAK_EVENT=1))
    monkeypatch.setattr(
        procs,
        "subprocess",
        SimpleNamespace(
            run=lambda args, **kw: calls.append(("taskkill", args)),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    proc = SimpleNamespace(pid=4242, wait=wait, tree_job=job)
    procs.kill_tree(proc, 0.01)  # type: ignore[arg-type]
    # the job object reaches descendants that outlive the leader, which taskkill /T cannot
    assert [name for name, _ in calls] == expected
    got = dict(calls)
    assert got["ctrl_break"] == (1, 4242)
    assert got["taskkill"] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert got.get("wait", 0.01) == 0.01
    assert got.get("terminate", (777, 1)) == (777, 1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_tree_waits_job_stop_s_between_term_and_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    waits: list[float | None] = []
    monkeypatch.setattr(procs.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(timeout: float | None = None) -> int:
        waits.append(timeout)
        raise subprocess.TimeoutExpired("job", timeout or 0)

    procs.kill_tree(SimpleNamespace(pid=4242, wait=wait))  # type: ignore[arg-type]
    # a job's own soft stop (two ffmpeg waits plus cleanup) fits before the hard kill (#257)
    assert waits == [procs.JOB_STOP_S]
    assert procs.JOB_STOP_S > 2 * procs.SOFT_STOP_S
    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_tree_without_term_sends_only_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    monkeypatch.setattr(procs.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(timeout: float | None = None) -> int:
        raise AssertionError("no soft stop to wait for")

    procs.kill_tree(SimpleNamespace(pid=4242, wait=wait), term=False)  # type: ignore[arg-type]
    assert signals == [signal.SIGKILL]

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(procs.os, "killpg", gone)  # the group already exited: nothing to report
    procs.kill_tree(SimpleNamespace(pid=4242, wait=wait), term=False)  # type: ignore[arg-type]


def test_jobs_list_sends_the_log_of_the_running_and_the_newest_job_only(
    client: TestClient,
) -> None:
    """A744: the page polls the jobs every second while one runs and shows the log of the
    running job, else of the newest; the other kept jobs (up to 50) come without their log."""
    jobs = client.app.state.jobs

    def keep(status: str, cancelled: bool = False) -> ui.Job:
        job = ui.Job("p", "analyze")
        job.status, job.cancelled = status, cancelled
        job.note("INFO", f"{status} job")
        jobs.jobs[job.id] = job
        return job

    old, failed = keep("completed"), keep("failed")
    running = keep("running", cancelled=True)  # Cancel asked, the process not gone yet
    try:
        got = client.get("/api/jobs").json()["jobs"]
        assert [j["id"] for j in got] == [old.id, failed.id, running.id]
        assert ["logs" in j for j in got] == [False, False, True]
        assert got[2]["status"] == "cancelling"
        assert got[2]["logs"][0].endswith("INFO running job")
        assert got[1]["status"] == "failed" and got[1]["progress"] == {"total": None, "steps": []}
        running.status = "cancelled"  # it ended: the newest keeps its log, for the page to show
        got = client.get("/api/jobs").json()["jobs"]
        assert ["logs" in j for j in got] == [False, False, True]
        assert got[2]["logs"][0].endswith("INFO running job")
        assert running.view()["logs"] == got[2]["logs"]  # cancel and start still send it
    finally:
        running.status = "cancelled"


@pytest.mark.parametrize(
    ("done", "cancelled", "expected"),
    [
        (True, True, []),  # exited by itself while stopping: no kill at all
        (False, True, [False]),  # still stuck after JOB_STOP_S: KILL only, no second TERM
        (False, False, [True]),  # not yet stopped: the full TERM then KILL, as before
    ],
)
def test_jobs_shutdown_does_not_term_a_cancelling_job_twice(
    monkeypatch: pytest.MonkeyPatch, done: bool, cancelled: bool, expected: list[bool]
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(procs, "kill_tree", lambda proc, wait=0, *, term=True: calls.append(term))
    monkeypatch.setattr(procs, "JOB_STOP_S", 0.01)
    jobs = ui.Jobs()
    job = ui.Job("p", "analyze")
    job.proc = SimpleNamespace(pid=4242)  # type: ignore[assignment]
    job.cancelled = cancelled
    if done:
        job.done.set()
    jobs.jobs[job.id] = job
    jobs.shutdown()
    assert calls == expected
    assert job.cancelled


def _linux_dead(pid: int) -> bool:
    """``pid`` is gone, or a zombie (dead, not yet reaped)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def _pid_alive(pid: int) -> bool:
    import ctypes  # noqa: PLC0415 - Windows only

    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return k32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102  # WAIT_TIMEOUT
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="Windows job objects and CTRL_BREAK")
def test_windows_cancel_ends_a_grandchild_that_ignores_ctrl_break() -> None:
    grandchild = (
        "import signal, time\n"
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        # longer than PROC_WAIT_S: a child left alive cannot pass the death check by exiting
        "time.sleep(600)\n"
    )
    leader = (
        "import subprocess, sys, time\n"
        f"c = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(600)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", leader])
    child = 0
    try:
        # two interpreter starts can be slow under a loaded suite
        wait_until(
            lambda: job.status != "running" or "ready" in list(job.logs),
            PROC_WAIT_S,
            "the grandchild's ready line",
        )
        logs = list(job.logs)
        assert "ready" in logs, (job.status, job.error, logs)
        child = int(next(line for line in logs if line.startswith("child")).split()[1])
        cancelled = jobs.cancel(job.id)
        # the leader dies on the event at once (only the grandchild ignores it), so the job
        # object kill follows promptly; JOB_STOP_S bounds the wait, plus time for the watcher
        # to see EOF
        wait_until(
            lambda: cancelled.status == "cancelled",
            procs.JOB_STOP_S + PROC_WAIT_S,
            "the job to be cancelled",
        )
        wait_until(lambda: not _pid_alive(child), PROC_WAIT_S, f"the grandchild {child} to die")
    finally:
        if child and _pid_alive(child):
            subprocess.run(["taskkill", "/PID", str(child), "/T", "/F"], capture_output=True)
        if job.proc is not None and job.proc.poll() is None:
            procs.kill_tree(job.proc)


def test_assemble_ignores_a_hidden_part_file_left_by_a_killed_export(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / ".001_a.part.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 400
    assert "export clips before assembling" in res.text


@pytest.mark.slow
@needs_ffmpeg
def test_a_timeline_only_assembly_succeeds_without_a_video_and_is_listed(
    client: TestClient, tmp_path: Path
) -> None:
    """#519: Assemble with an fcpxml format writes only the timeline, so the job completes
    without a final video and the timeline is listed as the export's own final."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    record_export(p, None)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=2:s=320x240:r=25",
         "-f", "lavfi", "-i", "sine=d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(p / "clips_ui_1" / "001_a.mp4")],
        check=True,
    )  # fmt: skip
    body = {"action": "assemble", "format": "fcpxml", "loudnorm": False}
    res = client.post(f"/api/projects/{pid}/jobs", json=body)
    assert res.status_code == 200, res.text
    job = wait_job(client)
    assert job["status"] == "completed", job
    got = client.get(f"/api/projects/{pid}").json()
    assert [o["name"] for o in got["outputs"]] == ["final_ui_1.fcpxml"]
    assert got["final_current"] is True
    assert not list(p.glob("*.mp4"))


@pytest.mark.parametrize("body", [{"loudnorm": True}, {"captions": True}])
@pytest.mark.parametrize("fmt", ["fcpxml", "otio"])
def test_assemble_job_refuses_audio_processing_and_captions_with_a_timeline_format(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, fmt: str
) -> None:
    """#519: as the CLI does (exit 2), a timeline format refuses Normalize loudness and Burn in
    captions instead of starting a job that would fail."""
    body = {"action": "assemble", "format": fmt, "loudnorm": False, **body}
    res, captured = peaks_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert not captured


def test_a_timeline_only_final_is_listed_on_its_own_and_a_timeline_beside_its_video_is_not(
    client: TestClient, tmp_path: Path
) -> None:
    """#519: a timeline assembled without a video (``--format fcpxml``) is a final of its own,
    newest first; a timeline next to its .mp4 is listed after that video, not as another final."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    for d in ("clips_ui_1", "clips_ui_2"):
        (p / d).mkdir()
        cache.save_json(p / d / "manifest.json", {"clips": []})
    exports = [{"dir": "clips_ui_1", "final": "final_ui_1.fcpxml"}, {"dir": "clips_ui_2"}]
    cache.save_json(p / "ui" / "exports.json", {"exports": exports})  # clips_ui_2: not assembled
    touch(p / "final.mp4", 300)
    touch(p / "final_ui_1.fcpxml", 100)
    touch(p / "final.fcpxml", 10)  # newest, but it belongs to final.mp4
    got = client.get(f"/api/projects/{pid}").json()
    assert [o["name"] for o in got["outputs"]] == ["final_ui_1.fcpxml", "final.mp4", "final.fcpxml"]
    assert got["final_current"] is False


@pytest.mark.slow
@needs_ffmpeg
def test_export_keeps_saved_order_and_each_assembly_is_kept(
    client: TestClient, tmp_path: Path
) -> None:
    """Review order is the clip order; re-exporting or re-assembling never overwrites."""
    video = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=8:s=320x240:r=25",
         "-f", "lavfi", "-i", "sine=d=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )  # fmt: skip
    pid, folder = make_project(client, tmp_path, {"src.mp4": video.read_bytes()})
    src = str((folder / "src.mp4").resolve())
    segments = [
        {"id": "s-2", "source": src, "start": 5, "end": 7, "label": "late"},
        {"id": "s-1", "source": src, "start": 0.5, "end": 2.5, "label": "early"},
    ]
    assert client.put(f"/api/projects/{pid}/review", json={"segments": segments}).status_code == 200

    def run(body: dict) -> dict:
        res = client.post(f"/api/projects/{pid}/jobs", json=body)
        assert res.status_code == 200, res.text
        job = wait_job(client)
        assert job["status"] == "completed", job
        return client.get(f"/api/projects/{pid}").json()

    project = run({"action": "clips"})
    assert [(c["id"], c["start"], c["file"][:3]) for c in project["clips"]["clips"]] == [
        ("s-2", 5.0, "001"),
        ("s-1", 0.5, "002"),
    ]
    first_dir = project["clips"]["dir"]
    assert project["outputs"] == []

    taken = ui.project_dir(pid) / f"final{first_dir[5:]}.ass"  # a caption file already there
    taken.write_text("keep me", encoding="utf-8")
    first = run({"action": "assemble"})["outputs"]
    assert taken.read_text(encoding="utf-8") == "keep me"
    assert first[0]["name"] != f"{taken.stem}.mp4"
    second = run({"action": "assemble", "transition": "cut"})["outputs"]
    assert first[0]["name"].endswith(".mp4") and first[0]["name"] != second[0]["name"]
    new = [o["name"] for o in second if o["name"] not in {x["name"] for x in first}]
    assert new[0].endswith(".mp4")  # assemble writes no timelines by default
    assert {n.rsplit(".", 1)[1] for n in new} <= {"mp4", "srt"}
    assert first[0]["name"] in [o["name"] for o in second]  # the earlier final stays listed
    assert (ui.project_dir(pid) / first[0]["name"]).is_file()  # the earlier final survives
    assert client.get(second[0]["url"]).status_code == 200

    again = run({"action": "clips"})  # fresh folder; the earlier export stays untouched
    assert again["clips"]["dir"] != first_dir
    # Every final stays listed (newest first), but none is the new export's own yet.
    assert again["outputs"] == second and again["final_current"] is False
    assert (ui.project_dir(pid) / first_dir / "manifest.json").is_file()


@pytest.fixture
def still_source(client: TestClient, tmp_path: Path) -> tuple[str, str, Path, Path]:
    """A project with one 1 s clip: (project id, source id, source file, project's frames dir)."""
    clip = tmp_path / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=320x240:r=25", "-c:v", "mpeg4", str(clip))
    pid, folder = make_project(client, tmp_path, {"a.mp4": clip.read_bytes()})
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    return pid, sid, folder / "a.mp4", ui.project_dir(pid) / ui.UI_DIR / "frames"


@needs_ffmpeg
@pytest.mark.parametrize(
    ("pid", "sid", "query", "token", "status", "cached"),
    [
        ("ok", "ok", "at=0.5", False, 403, None),
        ("nope", "ok", "at=0.5", True, 404, None),
        ("ok", "0" * 12, "at=0.5", True, 404, None),
        ("ok", "ok", "at=nan", True, 400, None),
        ("ok", "ok", "at=-1", True, 400, None),
        ("ok", "ok", "at=soon", True, 422, None),
        ("ok", "ok", "at=0.5&w=1", True, 200, "-50-32.jpg"),
        ("ok", "ok", "at=0.5&w=99999", True, 200, "-50-640.jpg"),
        ("ok", "ok", "at=0.5", True, 200, "-50-160.jpg"),
        ("ok", "ok", "at=99", True, 200, "-90-160.jpg"),  # clamped to duration - 0.1 s
        ("ok", "ok", "at=1e308", True, 200, "-90-160.jpg"),
    ],
)
def test_frame_endpoint_validates_input_and_clamps(
    client: TestClient,
    still_source: tuple[str, str, Path, Path],
    pid: str,
    sid: str,
    query: str,
    token: bool,
    status: int,
    cached: str | None,
) -> None:
    real_pid, real_sid, _, frames = still_source
    pid, sid = (real_pid if pid == "ok" else pid), (real_sid if sid == "ok" else sid)
    suffix = f"&t={client.headers['X-Cutpoint-Token']}" if token else ""
    res = client.get(
        f"/api/projects/{pid}/sources/{sid}/frame?{query}{suffix}",
        headers={"X-Cutpoint-Token": ""},
    )
    assert res.status_code == status, res.text
    if cached:
        assert [f.name[-len(cached) :] for f in frames.glob("*.jpg")] == [cached]
    else:
        assert not frames.exists()


@needs_ffmpeg
def test_frame_is_a_cached_jpeg_inside_the_project_and_the_source_is_untouched(
    client: TestClient, still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, source, frames = still_source
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    url = f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={client.headers['X-Cutpoint-Token']}"
    calls: list[float] = []
    real_grab = ui.grab_images
    patch_ui(
        monkeypatch, "grab_images", lambda *a, **k: (calls.append(a[1]), real_grab(*a, **k))[1]
    )

    first = client.get(url, headers={"X-Cutpoint-Token": ""})
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/jpeg"
    assert first.content[:3] == b"\xff\xd8\xff"
    assert "max-age=86400" in first.headers["cache-control"]
    files = list(frames.iterdir())
    assert [f.suffix for f in files] == [".jpg"]  # no .part left behind
    assert files[0].read_bytes() == first.content
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before

    second = client.get(url, headers={"X-Cutpoint-Token": ""})
    assert second.content == first.content
    assert calls == [0.5]  # ffmpeg ran once; the second request was served from the cache


@needs_ffmpeg
def test_frame_failure_is_a_500_and_leaves_no_cache_file(
    client: TestClient, still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _, frames = still_source

    def broken(*_a: object, **_k: object) -> list[bytes]:
        raise errors.FFmpegError("no image")

    patch_ui(monkeypatch, "grab_images", broken)
    res = client.get(
        f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={client.headers['X-Cutpoint-Token']}",
        headers={"X-Cutpoint-Token": ""},
    )
    assert res.status_code == 500
    assert not list(frames.glob("*")) if frames.exists() else True


@needs_ffmpeg
def test_frame_builds_one_still_at_a_time_even_after_a_failed_build(
    still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _, source, frames = still_source
    media = ui.Media()
    first_in, release_first, second_in, release_second = (threading.Event() for _ in range(4))
    calls: list[int] = []
    active = [0, 0]  # now, most at once
    arrived = threading.Event()  # a later request waits for the build, or builds alongside

    class Contended:
        """The still's build lock; tells when a request has to wait for it."""

        def __init__(self) -> None:
            self.lock = threading.Lock()

        def __enter__(self) -> None:
            if not self.lock.acquire(blocking=False):
                arrived.set()
                self.lock.acquire()

        def __exit__(self, *_exc: object) -> None:
            self.lock.release()

    class Locks(dict):
        def get(self, name, default=None):
            return super().get(name, (spy, 0))

    spy = Contended()
    media.frame_locks = Locks()

    def grab(*_a: object, **_k: object) -> list[bytes]:
        calls.append(1)
        if len(calls) > 1:
            arrived.set()
        active[0] += 1
        active[1] = max(active)
        try:
            if len(calls) == 1:
                first_in.set()
                release_first.wait(5)
                raise errors.FFmpegError("no image")
            if len(calls) == 2:
                second_in.set()
                release_second.wait(5)
            return [b"\xff\xd8\xffjpeg"]
        finally:
            active[0] -= 1

    patch_ui(monkeypatch, "grab_images", grab)
    results: list[object] = []

    def request() -> None:
        try:
            results.append(media.frame(ui.project_dir(pid), str(source), 0.5, 160))
        except errors.FFmpegError as e:
            results.append(e)

    a = threading.Thread(target=request)
    a.start()
    assert first_in.wait(5)
    b = threading.Thread(target=request)  # waits for the first builder's lock
    b.start()
    assert arrived.wait(5)
    release_first.set()  # the first build fails; the waiting request builds next
    assert second_in.wait(5)
    arrived.clear()
    c = threading.Thread(target=request)  # arrives mid-build: must wait, not build alongside
    c.start()
    assert arrived.wait(5)
    release_second.set()
    for t in (a, b, c):
        t.join(5)
    assert active[1] == 1 and len(calls) == 2
    assert sum(isinstance(r, Path) for r in results) == 2
    assert [f.suffix for f in frames.iterdir()] == [".jpg"]
    assert media.frame_locks == {}


@needs_ffmpeg
def test_frame_cache_drops_stale_stills_and_leftovers_once_per_run(
    client: TestClient, still_source: tuple[str, str, Path, Path]
) -> None:
    pid, sid, source, frames = still_source
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    key = ui.Media.key(source.resolve())
    frames.mkdir(parents=True)
    kept = frames / f"{key}-20-160.jpg"
    stale = frames / f"{'0' * 16}-20-160.jpg"  # a source that changed or left the project
    part = frames / f"{key}-30-160.123.part"
    other = frames.parent / "keep.txt"  # outside ui/frames: never touched
    for f in (kept, stale, part, other):
        f.write_bytes(b"x")
    url = f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={client.headers['X-Cutpoint-Token']}"

    assert client.get(url, headers={"X-Cutpoint-Token": ""}).status_code == 200
    assert sorted(f.name for f in frames.iterdir()) == sorted([kept.name, f"{key}-50-160.jpg"])
    assert other.is_file()
    stale.write_bytes(b"x")  # pruned once per run: a later request leaves it
    assert client.get(url, headers={"X-Cutpoint-Token": ""}).status_code == 200
    assert stale.is_file()
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


@needs_ffmpeg
def test_frame_cache_evicts_the_least_recently_used_beyond_the_cap(
    still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _, source, frames = still_source
    media = ui.Media()
    project = ui.project_dir(pid)
    patch_ui(monkeypatch, "FRAME_CACHE_MAX", 2)
    patch_ui(monkeypatch, "grab_images", lambda *_a, **_k: [b"\xff\xd8\xffjpeg"])
    first = media.frame(project, str(source), 0.1, 160)
    second = media.frame(project, str(source), 0.2, 160)
    old = time.time() - 100
    os.utime(first, (old, old))
    os.utime(second, (old + 1, old + 1))

    assert media.frame(project, str(source), 0.1, 160) == first  # a hit: now the newest
    third = media.frame(project, str(source), 0.3, 160)
    assert sorted(frames.iterdir()) == sorted([first, third])  # the least recently used went
    assert not second.exists()


def test_keyframe_grab_skips_to_the_keyframe_before_the_input() -> None:
    """#261: the filmstrip's stills decode only the keyframe at or before the time; the exact
    stills of the sequence thumbnails keep their accurate seek."""
    exact = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, quality=5)
    key = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, quality=5, keyframe=True)
    assert "-skip_frame" not in exact and "-noaccurate_seek" not in exact
    assert "-fps_mode" not in exact
    flags = ["-skip_frame", "nokey", "-noaccurate_seek"]
    i = key.index("-skip_frame")
    assert key[i : i + 3] == flags and i < key.index("-ss") < key.index("-i")
    # the keyframe lands before the seek point (a negative timestamp): passthrough keeps it
    j = key.index("-fps_mode")
    assert key[j : j + 2] == ["-fps_mode", "passthrough"] and key.index("-i") < j
    assert key[:i] + key[i + 3 : j] + key[j + 2 :] == exact


@needs_ffmpeg
def test_frame_key_mode_asks_for_the_keyframe_and_is_cached_apart_from_exact_stills(
    client: TestClient, still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _, frames = still_source
    base = f"/api/projects/{pid}/sources/{sid}/frame?t={client.headers['X-Cutpoint-Token']}"
    calls: list[tuple[object, object]] = []

    def grab(*a: object, **k: object) -> list[bytes]:
        calls.append((a[1], k["keyframe"]))
        return [b"\xff\xd8\xff" + (b"key" if k["keyframe"] else b"exact")]

    def get(query: str):
        return client.get(f"{base}&{query}", headers={"X-Cutpoint-Token": ""})

    patch_ui(monkeypatch, "grab_images", grab)
    key = get("at=0.5&key=1")
    assert key.status_code == 200 and key.content.endswith(b"key")
    exact = get("at=0.5")  # a keyframe still is never served for an exact one
    assert exact.status_code == 200 and exact.content.endswith(b"exact")
    assert get("at=0.5&key=1").content == key.content  # cached: ffmpeg is not run again
    assert calls == [(0.5, True), (0.5, False)]
    assert sorted(f.name.split("-", 1)[1] for f in frames.glob("*.jpg")) == [
        "50-160-k.jpg",
        "50-160.jpg",
    ]
    assert get("at=0.5&key=maybe").status_code == 422


@needs_ffmpeg
def test_a_real_keyframe_still_is_the_keyframe_at_or_before_the_time(
    client: TestClient, still_source: tuple[str, str, Path, Path]
) -> None:
    """The 1 s mpeg4 clip has a keyframe every 12 frames (0, 0.48 and 0.96 s at 25 fps): a
    keyframe still at 0.7 s is the frame at 0.48 s, not the one at 0.7 s nor the next keyframe
    (the default fps mode drops the keyframe's negative timestamp and writes the next one)."""
    pid, sid, source, _ = still_source
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    base = f"/api/projects/{pid}/sources/{sid}/frame?t={client.headers['X-Cutpoint-Token']}"

    def get(query: str):
        return client.get(f"{base}&{query}", headers={"X-Cutpoint-Token": ""})

    key = get("at=0.7&key=1")
    assert key.status_code == 200 and key.content[:3] == b"\xff\xd8\xff"
    assert key.content == get("at=0.48").content
    assert key.content != get("at=0.7").content
    assert key.content != get("at=0.96").content
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_a_huge_still_time_of_a_source_of_unknown_length_is_cut_not_an_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``at=1e308`` is finite, so the endpoint accepts it; with no known length to clamp to,
    its centiseconds overflowed (OverflowError, an HTTP 500). It is cut to FRAME_MAX_AT."""
    source = write(tmp_path / "a.mp4", b"x" * 10)
    project = tmp_path / "project"
    project.mkdir()
    patch_ui(monkeypatch, "probe", lambda _path: media_info(source, duration=0.0, **SD))
    starts: list[object] = []
    patch_ui(
        monkeypatch, "grab_images", lambda *a, **_k: (starts.append(a[1]), [b"\xff\xd8\xffjpeg"])[1]
    )
    media = ui.Media()
    still = media.frame(project, str(source), 1e308, 160)
    cs = round(ui.FRAME_MAX_AT * 100)
    assert still is not None and still.name == f"{media.key(source)}-{cs}-160.jpg"
    assert starts == [cs / 100]


STILL_BT2020 = {"color_primaries": "bt2020", "color_space": "bt2020nc", "pix_fmt": "yuv420p10le"}


TO_JPEG_709 = "colorspace=all=bt470bg:iall=bt709:range=pc:fast=1"
SD = {"width": 640, "height": 480}


@pytest.mark.parametrize(
    ("over", "kind"),
    [
        ({"vcodec": "hevc", "color_transfer": "smpte2084", **STILL_BT2020}, "hdr"),
        ({"vcodec": "av1", "color_transfer": "arib-std-b67", **STILL_BT2020}, "hdr"),
        ({"vcodec": "hevc"}, "bt709"),  # HD without a matrix tag is BT.709, as exports read it
        ({"color_transfer": "bt709", "color_primaries": "bt709", "color_space": "bt709"}, "bt709"),
        (SD, "bt601"),  # BT.601 (a JPEG already is): today's command and file name
        ({"color_space": "smpte170m", **SD}, "bt601"),
        ({"color_space": "bt470bg", "color_range": "pc", **SD}, "bt601"),
    ],
)
def test_still_is_converted_to_what_the_export_shows_and_cached_under_a_new_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over: dict[str, object], kind: str
) -> None:
    """#412, #441: a JPEG is BT.601 full range, so stills of HLG/PQ sources (an AV1 one the
    browser plays as is too) go through the export's sdr_filter and then to BT.601, and BT.709
    ones to BT.601; the filter's hash ends their name, so a still cached before is never served.
    BT.601 stills keep their command and name."""
    source = write(tmp_path / "a.mp4", b"x" * 10)
    project = tmp_path / "project"
    project.mkdir()
    info = media_info(source, duration=10.0, **over)
    patch_ui(monkeypatch, "probe", lambda _path: info)
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    calls: list[dict[str, object]] = []

    def grab(*_a: object, **k: object) -> list[bytes]:
        calls.append(k)
        return [b"\xff\xd8\xffjpeg"]

    patch_ui(monkeypatch, "grab_images", grab)
    media = ui.Media()
    still = media.frame(project, str(source), 0.5, 160)

    key = media.key(source)
    color = {
        "hdr": f"{encode.sdr_filter(info)},{TO_JPEG_709}",
        "bt709": TO_JPEG_709,
        "bt601": "",
    }[kind]
    assert color.startswith("zscale=") if kind == "hdr" else True
    assert [k["color"] for k in calls] == [color]
    tag = f"-{hashlib.sha1(color.encode()).hexdigest()[:8]}" if color else ""
    assert still is not None and still.name == f"{key}-50-160{tag}.jpg"
    media.prune_frames(project, [str(source)])  # the tagged name still belongs to its source
    assert still.is_file()


@pytest.mark.network
def test_ui_on_a_taken_port_fails_cleanly_without_printing_a_url(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn  # noqa: PLC0415

    monkeypatch.setattr(uvicorn.Server, "run", lambda *_a, **_k: pytest.fail("server started"))
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        with caplog.at_level(logging.ERROR):
            assert cli.main(["ui", "--port", str(port)]) == 1
    assert f"Port {port} is already in use on 127.0.0.1; pass --port <n>" in caplog.text
    assert "Traceback" not in caplog.text
    out = capsys.readouterr()
    assert "http://" not in out.out + out.err


def test_hangup_before_the_server_exists_exits_1() -> None:
    """SIGHUP before uvicorn runs unwinds through run()'s finally, which stops the children."""
    hangup = ui._Hangup()
    with pytest.raises(SystemExit) as info:
        hangup(1, None)
    assert info.value.code == 1 and hangup.received


def test_hangup_asks_a_running_server_to_stop_instead_of_raising() -> None:
    """uvicorn catches a SystemExit raised while it handles a request and keeps serving, so a
    running server is asked to stop; run() exits 1 once it has."""
    hangup = ui._Hangup()
    hangup.server = SimpleNamespace(should_exit=False)
    hangup(1, None)
    assert hangup.server.should_exit is True and hangup.received


@pytest.mark.slow
@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this OS")
def test_sighup_right_after_a_request_stops_the_ui_and_frees_its_port(tmp_path: Path) -> None:
    """Closing the terminal (SIGHUP) just after a request ends `cutpoint ui` with exit 1 and
    releases its port; a SystemExit from the handler used to be swallowed by uvicorn."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with (tmp_path / "ui.err").open("w") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "cutpoint.cli", "ui", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            cwd=tmp_path,
        )
    try:
        stdout = proc.stdout
        assert stdout is not None
        banner: list[str] = []  # read in a thread: a child stuck before its banner fails the test
        reader = threading.Thread(target=lambda: banner.append(stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout=PROC_WAIT_S)
        assert banner and banner[0].startswith("Cutpoint UI: http://127.0.0.1:"), banner

        def answers() -> bool:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).close()
            except urllib.error.HTTPError:
                pass  # any answer will do
            except OSError:
                return False
            return True

        # The banner comes just before the handler is installed; an answer comes after it.
        wait_until(answers, PROC_WAIT_S, "the server to answer")
        proc.send_signal(signal.SIGHUP)  # no idle wait: uvicorn may still be in the request
        try:
            code = proc.wait(timeout=PROC_WAIT_S)
        except subprocess.TimeoutExpired:
            pytest.fail(f"cutpoint ui still runs {PROC_WAIT_S:.0f} s after SIGHUP")
        assert code == 1, (tmp_path / "ui.err").read_text()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    ui.listen(port).close()  # the listener is closed: a new `cutpoint ui` can take the port


def test_show_folder_opens_only_the_projects_own_source_folder(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder comes from the project, never the request; token, 404 and 501 cases."""
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    assert client.get("/api/session").json()["can_reveal"] is True

    assert (
        client.post(f"/api/projects/{pid}/reveal", headers={"X-Cutpoint-Token": ""}).status_code
        == 403
    )
    assert opened == []
    # A path in the query or body is ignored: the project's folder is what opens.
    res = client.post(
        f"/api/projects/{pid}/reveal", params={"path": str(tmp_path)}, json={"path": str(tmp_path)}
    )
    assert res.status_code == 200, res.text
    assert opened == [folder] and res.json() == {"path": str(folder)}
    assert client.post("/api/projects/nope/reveal").status_code == 404
    assert client.post("/api/projects/..%2F..%2Fx/reveal").status_code != 200
    assert opened == [folder]

    # A source that is one video file opens its folder.
    meta = cache.load_json(ui.project_dir(pid) / "ui" / "meta.json")
    cache.save_json(
        ui.project_dir(pid) / "ui" / "meta.json", {**meta, "source": str(folder / "a.mp4")}
    )
    opened.clear()
    assert client.post(f"/api/projects/{pid}/reveal").status_code == 200
    assert opened == [folder]

    # A picked video that was deleted while its folder remains still opens that folder.
    cache.save_json(
        ui.project_dir(pid) / "ui" / "meta.json", {**meta, "source": str(folder / "gone.mp4")}
    )
    opened.clear()
    assert client.post(f"/api/projects/{pid}/reveal").status_code == 200
    assert opened == [folder]

    shutil.rmtree(folder)
    opened.clear()
    res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []

    folder.mkdir()
    patch_ui(monkeypatch, "folder_opener", lambda: None)
    assert client.get("/api/session").json()["can_reveal"] is False
    res = client.post(f"/api/projects/{pid}/reveal")
    assert (
        res.status_code == 501
        and res.json()["detail"] == "Opening folders isn't available on this machine"
    )


def test_show_folder_for_a_command_line_project_uses_its_recorded_sources(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    footage = tmp_path / "footage"
    for name in ("day1/a.mp4", "day2/b.mp4"):
        (footage / name).parent.mkdir(parents=True, exist_ok=True)
        (footage / name).write_bytes(b"x")
    p = ui.projects_dir() / "cli-proj"
    p.mkdir(parents=True)
    cache.save_json(
        p / "highlights.json",
        {
            "settings": {"inputs": [str(footage)]},
            "sources": [{"path": str(footage / n)} for n in ("day1/a.mp4", "day2/b.mp4")],
            "candidates": [],
        },
    )
    assert client.post("/api/projects/cli-proj/reveal").status_code == 200
    assert opened == [footage]


def test_show_folder_is_404_when_the_picked_folder_is_gone_or_unreadable(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deleted source folder is never swapped for its parent; OS errors are a 404, not a 500."""
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    meta = cache.load_json(ui.project_dir(pid) / "ui" / "meta.json")
    assert Path(meta["source"]) == folder

    def unreadable(real: Callable[..., bool]) -> Callable[..., bool]:
        def check(self: Path, *args: object, **kwargs: object) -> bool:
            if self == folder or folder in self.parents:
                raise PermissionError(13, "Permission denied", str(self))
            return real(self, *args, **kwargs)

        return check

    with monkeypatch.context() as m:
        for name in ("is_file", "is_dir", "exists"):
            m.setattr(Path, name, unreadable(getattr(Path, name)))
        res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []

    shutil.rmtree(folder)  # a picked folder that is gone is never swapped for its parent
    res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []


@pytest.fixture
def reveal_env(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Path], str, Path, Path, str]:
    """A project whose folder opener records what it opens.

    Returns (opened folders, project id, source folder, project folder, reveal URL).
    """
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    return opened, pid, folder, ui.project_dir(pid), f"/api/projects/{pid}/reveal"


def test_show_folder_targets_open_the_latest_clips_and_the_final_folder(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reveal_env: tuple[list[Path], str, Path, Path, str],
) -> None:
    """``target`` picks one of the project's folders; the path still never comes from the client."""
    opened, _pid, folder, p, reveal = reveal_env

    assert (
        client.post(
            reveal, params={"target": "clips"}, headers={"X-Cutpoint-Token": ""}
        ).status_code
        == 403
    )
    for target in ("clips", "outputs"):
        res = client.post(reveal, params={"target": target})
        assert res.status_code == 404, target
    for target in ("../x", str(tmp_path), "Clips", ""):
        assert client.post(reveal, params={"target": target}).status_code == 400, target
    assert opened == []

    # An export: clips opens its folder, outputs is still 404 until something is assembled.
    (p / "clips_vertical").mkdir()
    cache.save_json(p / "clips_vertical" / "manifest.json", {"clips": []})
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_vertical"}]})
    res = client.post(reveal, params={"target": "clips", "path": str(tmp_path)})
    assert res.status_code == 200 and res.json() == {"path": str(p / "clips_vertical")}
    assert opened == [p / "clips_vertical"]
    assert client.post(reveal, params={"target": "outputs"}).status_code == 404

    cache.save_json(
        p / "ui" / "exports.json", {"exports": [{"dir": "clips_vertical", "final": "final.mp4"}]}
    )
    (p / "final.mp4").write_bytes(b"x")
    opened.clear()
    res = client.post(reveal, params={"target": "outputs"}, json={"path": str(tmp_path)})
    assert res.status_code == 200 and res.json() == {"path": str(p)}
    assert opened == [p]

    # No target still means the sources, as before.
    opened.clear()
    assert client.post(reveal).status_code == 200
    assert client.post(reveal, params={"target": "sources"}).status_code == 200
    assert opened == [folder, folder]

    def broken(_: Path) -> None:
        raise OSError(5, "Access is denied")

    patch_ui(monkeypatch, "folder_opener", lambda: broken)
    res = client.post(reveal, params={"target": "outputs"})
    assert res.status_code == 500 and "Access is denied" in res.json()["detail"]


def test_show_folder_project_opens_the_project_folder_itself(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reveal_env: tuple[list[Path], str, Path, Path, str],
) -> None:
    """AI Chat's Open project folder (#1010): always there, nothing exported or assembled yet;
    the path still comes from the project, and a machine without a file manager says so."""
    opened, _pid, _folder, p, reveal = reveal_env
    res = client.post(reveal, params={"target": "project", "path": str(tmp_path)})
    assert res.status_code == 200 and res.json() == {"path": str(p)}
    assert opened == [p]
    assert (
        client.post(
            reveal, params={"target": "project"}, headers={"X-Cutpoint-Token": ""}
        ).status_code
        == 403
    )
    patch_ui(monkeypatch, "folder_opener", lambda: None)
    res = client.post(reveal, params={"target": "project"})
    assert res.status_code == 501
    assert res.json()["detail"] == "Opening folders isn't available on this machine"
    assert opened == [p]


@pytest.mark.parametrize("bad", ["..", ".", ".hidden"])
def test_exports_json_cannot_point_outside_the_project(
    client: TestClient, reveal_env: tuple[list[Path], str, Path, Path, str], bad: str
) -> None:
    opened, _pid, _folder, p, reveal = reveal_env
    # Manifests and videos where a "." / ".." / dot-leading name would resolve to.
    for folder in (p, p.parent, p / ".hidden"):
        folder.mkdir(exist_ok=True)
        cache.save_json(folder / "manifest.json", {"clips": []})
    (p / ".hidden.mp4").write_bytes(b"x")

    def write(*items: dict[str, str]) -> None:
        cache.save_json(p / "ui" / "exports.json", {"exports": list(items)})

    write({"dir": bad})
    assert client.post(reveal, params={"target": "clips"}).status_code == 404

    # A bad final is ignored; the good export stays usable.
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    write({"dir": "clips_ui_1", "final": ".hidden.mp4" if bad == ".hidden" else bad})
    assert client.post(reveal, params={"target": "outputs"}).status_code == 404

    # A bad entry after a legitimate one is dropped, not used as the current export.
    (p / "final.mp4").write_bytes(b"x")
    write({"dir": "clips_ui_1", "final": "final.mp4"}, {"dir": bad})
    assert client.post(reveal, params={"target": "clips"}).json() == {"path": str(p / "clips_ui_1")}
    assert client.post(reveal, params={"target": "outputs"}).json() == {"path": str(p)}
    assert opened == [p / "clips_ui_1", p]


@pytest.mark.parametrize("kind", ["dir", "final"])
def test_exports_json_cannot_point_through_a_link_outside_the_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    reveal = f"/api/projects/{pid}/reveal"
    outside = tmp_path / "outside"
    outside.mkdir()
    cache.save_json(outside / "manifest.json", {"clips": []})
    (outside / "final.mp4").write_bytes(b"x")
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "final.mp4").write_bytes(b"x")

    def write(*items: dict[str, str]) -> None:
        cache.save_json(p / "ui" / "exports.json", {"exports": list(items)})

    good = {"dir": "clips_ui_1", "final": "final.mp4"}
    if kind == "dir":
        link_dir(p / "clips_link", outside)
        bad = {"dir": "clips_link", "final": "final.mp4"}
        write(bad)
        assert client.post(reveal, params={"target": "clips"}).status_code == 404
    else:
        try:
            (p / "final_link.mp4").symlink_to(outside / "final.mp4")
        except (OSError, NotImplementedError):
            pytest.skip("cannot create a file symlink here")
        bad = {"dir": "clips_ui_1", "final": "final_link.mp4"}
        write(bad)

    def listed() -> list[str]:
        return [o["name"] for o in client.get(f"/api/projects/{pid}").json()["outputs"]]

    # dir: an earlier legitimate export takes over; final: the latest export stays current,
    # its linked final is never listed or served (the project's own final.mp4 still is).
    write(good, bad)
    if kind == "dir":
        assert client.post(reveal, params={"target": "clips"}).json() == {
            "path": str(p / "clips_ui_1")
        }
        assert opened == [p / "clips_ui_1"]
    else:
        assert listed() == ["final.mp4"]
        assert client.get(f"/api/projects/{pid}/outputs/final_link.mp4").status_code == 404
        write(good)
        assert listed() == ["final.mp4"]
        assert client.post(reveal, params={"target": "outputs"}).json() == {"path": str(p)}
        # Timelines next to the final are served only when they stay inside the project.
        (outside / "final.fcpxml").write_text("x")
        (p / "final.fcpxml").symlink_to(outside / "final.fcpxml")
        assert client.get(f"/api/projects/{pid}/outputs/final.fcpxml").status_code == 404
        assert listed() == ["final.mp4"]  # the linked timeline stays excluded


def test_export_manifest_symlinked_outside_the_project_is_not_served(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    outside = tmp_path / "outside"
    outside.mkdir()
    cache.save_json(outside / "manifest.json", {"clips": [{"label": "SECRET-OUTSIDE"}]})
    (p / "clips_ui_1").mkdir()
    try:
        (p / "clips_ui_1" / "manifest.json").symlink_to(outside / "manifest.json")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file symlink here")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200
    assert "SECRET-OUTSIDE" not in res.text


def test_fresh_name_treats_a_broken_symlink_as_taken(tmp_path: Path) -> None:
    """A dangling link is a taken name: nothing may be written through it."""
    try:
        (tmp_path / "final.mp4").symlink_to(tmp_path / "missing-target")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink here")
    assert ui.fresh_name(tmp_path, "final", ("", ".mp4", ".fcpxml")) == "final-2"


def test_folder_opener_runs_the_platform_tool_without_a_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[object, dict]] = []

    class FakeProc:
        def wait(self) -> int:
            return 0

    def popen(args, **kw):
        calls.append((args, kw))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    monkeypatch.setattr(sys, "platform", "linux")
    assert ui.folder_opener() is None  # no display: nobody would see the window
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    ui.folder_opener()(tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    ui.folder_opener()(tmp_path)
    assert [c[0] for c in calls] == [
        ["/usr/bin/xdg-open", str(tmp_path)],
        ["/usr/bin/open", str(tmp_path)],
    ]
    assert all(not kw.get("shell") for _, kw in calls)
    assert all(kw["stdout"] == kw["stderr"] == subprocess.DEVNULL for _, kw in calls)

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert ui.folder_opener() is None  # macOS without `open`, or Linux without xdg-open

    started: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", started.append, raising=False)
    ui.folder_opener()(tmp_path)
    assert started == [str(tmp_path)] and len(calls) == 2


# --------------------------------------------------------------------------- native picker
class FakeDialog:
    """A stand-in for the dialog process: fixed output, or blocks until released."""

    def __init__(self, out: str = "", code: int = 0, err: str = "") -> None:
        self.out, self.returncode, self.err = out, code, err
        self.calls: list[list[str]] = []
        self.opened = threading.Event()
        self.release = threading.Event()
        self.block = False

    def __call__(self, args, **kw):
        self.calls.append(args)
        outer = self

        class Proc:
            returncode = outer.returncode

            def communicate(self):
                outer.opened.set()
                if outer.block:
                    assert outer.release.wait(30)
                return outer.out, outer.err

            def kill(self):
                outer.release.set()

        return Proc()


@pytest.fixture
def dialog(monkeypatch: pytest.MonkeyPatch) -> FakeDialog:
    fake = FakeDialog()
    monkeypatch.setattr(subprocess, "Popen", fake)
    patch_ui(monkeypatch, "path_picker", lambda: lambda kind, start: (["dlg", kind], 1))
    return fake


def test_pick_returns_the_chosen_path_or_null_on_cancel(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
    assert client.get("/api/session").json()["can_pick"] is True
    video = tmp_path / "v.mp4"
    dialog.out = f"{video}\n"
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 200 and res.json() == {"path": os.path.normpath(video)}
    assert dialog.calls == [["dlg", "file"]]

    dialog.out = ""  # closed without choosing (exit 0, nothing printed)
    assert client.post("/api/pick", params={"kind": "folder"}).json() == {"path": None}
    dialog.returncode = 1  # the cancel exit code of zenity/kdialog
    assert client.post("/api/pick", params={"kind": "folder"}).json() == {"path": None}


def test_pick_errors(
    client: TestClient, dialog: FakeDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post("/api/pick", params={"kind": "other"}).status_code == 400
    assert dialog.calls == []

    dialog.returncode, dialog.err = 2, "boom"
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 500 and "failed" in res.json()["detail"]

    dialog.returncode, dialog.out = 0, "relative/path.mp4\n"
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 500

    def broken(*a, **kw):
        raise OSError(2, "No such file")

    monkeypatch.setattr(subprocess, "Popen", broken)
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 500 and "No such file" in res.json()["detail"]
    # The failure released the lock, so the next dialog can open.
    monkeypatch.setattr(subprocess, "Popen", dialog)
    dialog.out = ""
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200

    patch_ui(monkeypatch, "path_picker", lambda: None)
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 501
    assert client.get("/api/session").json()["can_pick"] is False


def test_pick_allows_one_dialog_at_a_time(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
    dialog.block, dialog.out = True, str(tmp_path / "v.mp4")
    first: list[int] = []
    t = threading.Thread(
        target=lambda: first.append(client.post("/api/pick", params={"kind": "file"}).status_code)
    )
    t.start()
    try:
        assert dialog.opened.wait(30)
        assert client.post("/api/pick", params={"kind": "folder"}).status_code == 409
    finally:
        dialog.release.set()
        t.join(30)
    assert first == [200] and len(dialog.calls) == 1
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200


def test_pick_start_is_the_nearest_existing_absolute_folder(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "f.mp4").write_bytes(b"x")
    assert ui.pick_start(str(tmp_path / "a")) == tmp_path / "a"
    assert ui.pick_start(str(tmp_path / "a" / "gone" / "deeper")) == tmp_path / "a"
    assert ui.pick_start(str(tmp_path / "f.mp4")) == tmp_path  # a file: its folder
    for ignored in ("", "relative/dir", "a", "bad\0path"):
        assert ui.pick_start(ignored) is None, ignored


def test_pick_passes_start_to_the_dialog(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[Path | None] = []

    def build(kind, start):
        starts.append(start)
        return ["dlg"], None

    patch_ui(monkeypatch, "path_picker", lambda: build)
    monkeypatch.setattr(subprocess, "Popen", FakeDialog())
    client.post("/api/pick", params={"kind": "file", "start": str(tmp_path / "missing" / "x")})
    client.post("/api/pick", params={"kind": "file", "start": "relative"})
    assert starts == [tmp_path, None]


def test_path_picker_builds_the_platform_dialog_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    present: set[str] = set()
    monkeypatch.setattr(shutil, "which", lambda n: f"/bin/{n}" if n in present else None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    present.update({"zenity", "kdialog"})
    assert ui.path_picker() is None  # no display: nobody would see the window
    monkeypatch.setenv("DISPLAY", ":0")
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:2] == ["/bin/zenity", "--file-selection"]
    assert "--directory" in argv and f"--filename={tmp_path}{os.sep}" in argv
    assert any(a.startswith("--title=") for a in argv)
    assert cancel == 1
    argv, _ = ui.path_picker()("file", None)
    assert argv[:2] == ["/bin/zenity", "--file-selection"] and "--directory" not in argv
    assert any("*.mp4" in a for a in argv) and not any(a.startswith("--filename") for a in argv)
    assert any(a.startswith("--file-filter=Insta360") and "*.insv" in a for a in argv)

    present.discard("zenity")  # kdialog is the fallback
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[0] == "/bin/kdialog" and "--title" in argv
    assert argv[argv.index("--getexistingdirectory") + 1] == str(tmp_path)
    argv, cancel = ui.path_picker()("file", tmp_path)
    assert argv[0] == "/bin/kdialog" and "--getexistingdirectory" not in argv
    assert argv[argv.index("--getopenfilename") + 1] == str(tmp_path)
    assert "*.mp4" in argv[-1] and cancel == 1

    present.clear()
    assert ui.path_picker() is None  # a display but no dialog tool

    monkeypatch.setattr(sys, "platform", "darwin")
    assert ui.path_picker() is None
    present.add("osascript")
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:2] == ["/bin/osascript", "-e"] and "choose" in argv[2]
    assert argv[3:] == ["folder", str(tmp_path)] and cancel is None
    assert ui.path_picker()("file", None)[0][3:] == ["file"]

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert ui.path_picker() is None  # Python without tkinter
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:3] == [sys.executable, "-I", "-c"]
    assert argv[4:6] == ["folder", str(tmp_path)] and "*.mp4" in argv[6] and cancel is None
    assert ui.path_picker()("file", None)[0][5] == ""


@pytest.mark.parametrize(("platform", "var"), [("win32", "SSH_CONNECTION"), ("darwin", "SSH_TTY")])
def test_path_picker_is_off_in_an_ssh_session_on_windows_and_macos(
    monkeypatch: pytest.MonkeyPatch, platform: str, var: str
) -> None:
    monkeypatch.setattr(shutil, "which", lambda n: f"/bin/{n}")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    assert ui.path_picker() is not None
    monkeypatch.setenv(
        var, "10.0.0.1 22 10.0.0.2 5555" if var == "SSH_CONNECTION" else "/dev/pts/0"
    )
    assert ui.path_picker() is None


def test_picker_shutdown_kills_an_open_dialog_and_frees_the_lock(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
    dialog.block = True
    killed: list[bool] = []
    real_kill = dialog.release.set
    dialog.release.set = lambda: (killed.append(True), real_kill())[1]  # type: ignore[method-assign]
    status: list[int] = []
    t = threading.Thread(
        target=lambda: status.append(client.post("/api/pick", params={"kind": "file"}).status_code)
    )
    t.start()
    try:
        assert dialog.opened.wait(30)
        picker = client.app.state.picker
        assert picker.proc is not None
        picker.shutdown()
    finally:
        dialog.release.set()
        t.join(30)
    assert not t.is_alive() and killed and picker.proc is None
    dialog.block, dialog.out = False, ""
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200


# --------------------------------------------------------------------------- settings
def test_settings_report_the_folder_the_lock_and_the_config_file(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    res = client.get("/api/settings")
    assert res.status_code == 200
    got = res.json()
    assert got["projects_dir"] == str(config.projects_dir())
    assert got["projects_dir_locked"] is False
    assert got["config_file"] == str(home / "config.json")
    assert got["vision"] is False and got["vision_model"]
    locked = tmp_path / "locked"
    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(locked))
    config.get_config.cache_clear()
    got = client.get("/api/settings").json()
    assert got["projects_dir_locked"] is True and got["projects_dir"] == str(locked)


def test_put_settings_saves_the_folder_and_lists_projects_from_it(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    new = tmp_path / "elsewhere"  # only the last part may be created
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 200, res.text
    assert res.json()["projects_dir"] == str(new) and new.is_dir()
    assert json.loads((home / "config.json").read_text("utf-8")) == {"projects_dir": str(new)}
    assert client.get("/api/settings").json()["projects_dir"] == str(new)

    (new / "trip").mkdir()
    cache.save_json(new / "trip" / "highlights.json", {})
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == ["trip"]

    other = tmp_path / "other"  # an existing folder works too
    other.mkdir()
    assert client.put("/api/settings", json={"projects_dir": f" {other} "}).status_code == 200
    assert client.get("/api/projects").json() == {"projects": []}


def test_put_settings_rejects_unusable_folders(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    cases = [
        {},
        {"projects_dir": ""},
        {"projects_dir": 5},
        {"projects_dir": "relative/dir"},
        {"projects_dir": str(a_file)},
        {"projects_dir": str(tmp_path / "no" / "parent")},
    ]
    for body in cases:
        assert client.put("/api/settings", json=body).status_code == 400, body
    assert not (tmp_path / "no").exists()
    assert not (home / "config.json").exists()


def test_put_settings_conflicts(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"projects_dir": str(tmp_path)}
    cfg = home / "config.json"

    assert client.get("/api/settings").status_code == 200  # config loaded (and cached) while valid
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json", encoding="utf-8")
    new = tmp_path / "fresh"
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 409 and cfg.read_text("utf-8") == "{not json"  # never overwritten
    assert not new.exists()  # nor is the new folder created
    cfg.unlink()
    config.get_config.cache_clear()

    client.app.state.jobs.jobs["j1"] = SimpleNamespace(status="running", proc=None)
    assert client.put("/api/settings", json=body).status_code == 409
    client.app.state.jobs.jobs["j1"].status = "done"
    assert client.put("/api/settings", json=body).status_code == 200

    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(tmp_path / "env"))
    res = client.put("/api/settings", json={"projects_dir": str(tmp_path / "x")})
    assert res.status_code == 409 and "CUTPOINT_PROJECTS_DIR" in res.json()["detail"]
    assert json.loads(cfg.read_text("utf-8")) == {"projects_dir": str(tmp_path)}


def test_put_settings_saves_the_vision_default(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{\n  "player_tags": ["Me"]\n}', encoding="utf-8")
    res = client.put("/api/settings", json={"vision": True})
    assert res.status_code == 200, res.text
    assert res.json()["vision"] is True
    assert json.loads(cfg.read_text("utf-8")) == {"player_tags": ["Me"], "vision": True}
    assert client.get("/api/settings").json()["vision"] is True
    assert config.get_config().vision is True

    # Unticking removes the key (the default) instead of writing false (#919).
    assert client.put("/api/settings", json={"vision": False}).json()["vision"] is False
    assert config.get_config().vision is None
    assert json.loads(cfg.read_text("utf-8")) == {"player_tags": ["Me"]}
    assert client.put("/api/settings", json={"vision": True}).json()["vision"] is True

    both = {"projects_dir": str(tmp_path), "vision": True}
    got = client.put("/api/settings", json=both).json()
    assert got["vision"] is True and got["projects_dir"] == str(tmp_path)
    data = json.loads(cfg.read_text("utf-8"))
    assert data == {"player_tags": ["Me"], "vision": True, "projects_dir": str(tmp_path)}

    other = tmp_path / "other"  # a later folder replaces the old one in place; other keys stay
    other.mkdir()
    assert client.put("/api/settings", json={"projects_dir": str(other)}).status_code == 200
    data = json.loads(cfg.read_text("utf-8"))
    assert list(data) == ["player_tags", "vision", "projects_dir"]
    assert data["projects_dir"] == str(other)


def test_put_settings_vision_needs_a_boolean_and_no_folder_rules(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for value in ("yes", 1, None):
        assert client.put("/api/settings", json={"vision": value}).status_code == 400, value
    assert not (home / "config.json").exists()

    # Only the projects folder is locked by CUTPOINT_PROJECTS_DIR or a running job.
    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(tmp_path / "env"))
    client.app.state.jobs.jobs["j1"] = SimpleNamespace(status="running", proc=None, cancelled=False)
    assert client.put("/api/settings", json={"vision": True}).status_code == 200
    both = {"vision": False, "projects_dir": str(tmp_path)}
    assert client.put("/api/settings", json=both).status_code == 409
    assert json.loads((home / "config.json").read_text("utf-8")) == {"vision": True}


def test_settings_show_a_wrongly_typed_vision_as_off_and_replace_it(
    client: TestClient, home: Path
) -> None:
    set_config(home, vision="yes")
    assert client.get("/api/projects").status_code == 409  # rejected by get_config
    assert client.get("/api/settings").json()["vision"] is False
    assert client.put("/api/settings", json={"vision": True}).status_code == 200
    assert client.get("/api/projects").status_code == 200


@pytest.mark.parametrize(
    ("cfg", "shown"),
    [(None, "off"), ("light", "light"), ("strong", "strong"), ("loud", "off"), (3, "off")],
)
def test_settings_show_the_compress_peaks_default(
    client: TestClient, home: Path, cfg: object, shown: str
) -> None:
    """The page's Compress peaks select starts on config.json's value; an invalid one shows off."""
    if cfg is not None:
        set_config(home, compress_peaks=cfg)
    assert client.get("/api/settings").json()["compress_peaks"] == shown


def peaks_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict
) -> tuple[object, list[list[str]]]:
    """Start ``body``'s job on a command-line project that has a review and an export."""
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / "001_a.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    captured = JobSpy(monkeypatch, check=False).args
    return client.post("/api/projects/cli/jobs", json=body), captured


@pytest.mark.parametrize("action", ["clips", "assemble"])
@pytest.mark.parametrize(
    ("body", "cfg", "passed"),
    [
        ({"compress_peaks": "strong"}, None, "strong"),
        ({"compress_peaks": "light"}, "strong", "light"),
        ({"compress_peaks": "off"}, "strong", "off"),
        ({}, "light", "light"),  # not sent: config.json's value, explicitly
        ({}, None, "off"),
    ],
)
def test_clips_and_assemble_jobs_pass_compress_peaks(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
    cfg: str | None,
    passed: str,
) -> None:
    if cfg is not None:
        set_config(home, compress_peaks=cfg)
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": action, **body})
    assert res.status_code == 200, res.text
    args = captured[0]
    assert args.count("--compress-peaks") == 1
    assert args[args.index("--compress-peaks") + 1] == passed


@pytest.mark.parametrize("action", ["clips", "assemble"])
@pytest.mark.parametrize("value", ["loud", "", None, True, ["light"]])
def test_clips_and_assemble_jobs_reject_an_unknown_compress_peaks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str, value: object
) -> None:
    body = {"action": action, "compress_peaks": value}
    res, captured = peaks_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert "compress_peaks must be one of off, light, strong" in res.json()["detail"]
    assert not captured


def test_put_settings_removes_a_folder_it_created_when_saving_fails(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new = tmp_path / "fresh"

    def broken(*a, **kw):
        raise OSError(28, "No space left")

    patch_ui(monkeypatch, "update_config", broken)
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 500 and not new.exists()


def test_put_settings_never_removes_an_existing_folder_on_failure(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "keep"
    existing.mkdir()
    (existing / "f.txt").write_text("x")

    def broken(*a, **kw):
        raise OSError(28, "No space left")

    patch_ui(monkeypatch, "update_config", broken)
    res = client.put("/api/settings", json={"projects_dir": str(existing)})
    assert res.status_code == 500 and (existing / "f.txt").exists()


def test_check_projects_dir_removes_its_own_folder_when_the_write_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    new = tmp_path / "probe"
    monkeypatch.setattr(tempfile, "TemporaryFile", denied)
    with pytest.raises(HTTPException) as e:
        ui.check_projects_dir(str(new))
    assert e.value.status_code == 400 and "Permission denied" in e.value.detail
    assert not new.exists()
    existing = tmp_path / "keep"  # a folder that was already there is never removed
    existing.mkdir()
    (existing / "f.txt").write_text("x")
    with pytest.raises(HTTPException):
        ui.check_projects_dir(str(existing))
    assert (existing / "f.txt").exists()
    monkeypatch.undo()
    assert ui.check_projects_dir(str(new)) == (new, True)
    assert ui.check_projects_dir(str(new)) == (new, False)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("{oops", "Expecting property name"),
        ('{"video_encoder": "bogus"}', "video_encoder is 'bogus'"),
        ('{"source_remap_exclude": 5}', "int"),  # wrong type: ValueError -> 409, not a 500
        ('{"projects_dir": 5}', "int"),
    ],
)
def test_project_routes_name_a_rejected_config_file_and_settings_still_work(
    client: TestClient,
    home: Path,
    tmp_path: Path,
    text: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the env var would win over the bogus config value
    monkeypatch.delenv("CUTPOINT_VIDEO_ENCODER")
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text, encoding="utf-8")
    config.get_config.cache_clear()
    token = client.app.state.token
    calls = [
        ("GET", "/api/projects", {}),
        ("POST", "/api/projects", {"json": {"name": "x", "source": str(tmp_path / "footage")}}),
        ("GET", f"/api/projects/{pid}", {}),
        ("PUT", f"/api/projects/{pid}/review", {"json": {}}),
        ("PUT", f"/api/projects/{pid}/target", {"json": {"seconds": 60}}),
        ("POST", f"/api/projects/{pid}/reveal", {}),
        ("POST", f"/api/projects/{pid}/jobs", {"json": {"action": "analyze"}}),
        ("POST", f"/api/projects/{pid}/sources/s0/prepare", {}),
        ("GET", f"/api/projects/{pid}/sources/s0", {"params": {"t": token}}),
        ("GET", f"/api/projects/{pid}/sources/s0/frame", {"params": {"t": token}}),
        ("GET", f"/api/projects/{pid}/outputs/final.mp4", {"params": {"t": token}}),
    ]
    for method, path, kw in calls:
        res = client.request(method, path, **kw)
        assert res.status_code == 409, (method, path, res.text)
        detail = res.json()["detail"]
        assert str(cfg) in detail and error in detail, (path, detail)
    assert client.get("/api/projects", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    assert client.get("/api/session").status_code == 200
    assert client.get("/api/jobs").status_code == 200
    settings = client.get("/api/settings")
    if text.startswith("{oops"):  # unreadable: Settings say so too, and never overwrite it
        assert settings.status_code == 409 and str(cfg) in settings.json()["detail"]
    else:  # a rejected unrelated value: Settings still show and save the projects folder
        assert settings.status_code == 200 and settings.json()["config_file"] == str(cfg)
        if "projects_dir" in text:  # shown as typed, so it can be replaced
            assert settings.json()["projects_dir"] == "5"
        assert client.put("/api/settings", json={"projects_dir": str(tmp_path)}).status_code == 200
        # the rejected unrelated value is kept; a bad projects_dir is the one thing replaced
        assert json.loads(cfg.read_text("utf-8")) == {
            **json.loads(text),
            "projects_dir": str(tmp_path),
        }

    cfg.unlink()  # fixed by hand: the projects come back without a restart
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == [pid]


def test_a_rejected_encoder_variable_is_named_instead_of_config_json(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "bogus")
    config.get_config.cache_clear()
    res = client.get("/api/projects")
    assert res.status_code == 409
    detail = res.json()["detail"]
    assert detail.startswith("CUTPOINT_VIDEO_ENCODER is 'bogus'; expected one of auto,"), detail
    assert str(cfg) not in detail


@pytest.mark.parametrize(
    "content",
    [
        "{oops",
        '{"base": "funny", "weights": [1, 2]}',
        '{"base": "funny", "weights": {"x": null}}',
        '{"name": "broken", "base": "funny"}',
    ],
    ids=["invalid-json", "weights-not-object", "weight-null", "no-schema"],
)
def test_a_broken_learned_style_is_left_out_with_a_warning(
    client: TestClient, home: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    styles = home / "styles"
    styles.mkdir(parents=True)
    (styles / "broken.json").write_text(content, encoding="utf-8")
    mine = {"schema": 1, "name": "mine", "base": "funny", "source": "hand"}
    (styles / "mine.json").write_text(json.dumps(mine), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        res = client.get("/api/session")
    assert res.status_code == 200
    listed = res.json()["styles"]
    assert "mine" in listed and "funny" in listed and "broken" not in listed
    assert str(styles / "broken.json") in caplog.text


# ------------------------------------------------- settings change vs job start (#143)
SLEEP_JOB = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.mark.slow
def test_put_settings_refuses_a_job_that_started_during_the_check_and_removes_the_new_folder(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = client.app.state.jobs
    real_check = ui.check_projects_dir
    started: list = []

    def check_then_start_job(value):
        result = real_check(value)  # creates the folder
        started.append(jobs.start("p", "analyze", SLEEP_JOB))  # a second tab wins the race
        return result

    patch_ui(monkeypatch, "check_projects_dir", check_then_start_job)
    new = tmp_path / "fresh"
    try:
        res = client.put("/api/settings", json={"projects_dir": str(new)})
        assert res.status_code == 409 and "running job" in res.json()["detail"]
        assert not new.exists()
        assert not (home / "config.json").exists()
    finally:
        for job in started:
            jobs.cancel(job.id)


@pytest.mark.slow
def test_a_job_cannot_start_while_the_projects_folder_is_being_written(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = client.app.state.jobs
    real_write = ui.update_config
    attempt: dict = {}

    def write_while_another_tab_starts_a_job(change):
        def start() -> None:
            attempt["job"] = jobs.start("p", "analyze", SLEEP_JOB)
            attempt["wrote"] = wrote.is_set()
            done.set()

        wrote, done = threading.Event(), threading.Event()
        attempt["wrote_event"] = wrote
        threading.Thread(target=start, daemon=True).start()
        attempt["overlapped"] = done.wait(0.5)  # True: the job started mid-write
        data = real_write(change)
        wrote.set()
        return data

    patch_ui(monkeypatch, "update_config", write_while_another_tab_starts_a_job)
    new = tmp_path / "elsewhere"
    try:
        res = client.put("/api/settings", json={"projects_dir": str(new)})
        assert res.status_code == 200, res.text
        assert attempt["overlapped"] is False
        assert attempt["wrote_event"].wait(5)
    finally:
        deadline = time.time() + 5
        while "job" not in attempt and time.time() < deadline:
            time.sleep(0.02)
        if "job" in attempt:
            jobs.cancel(attempt["job"].id)
    assert attempt["wrote"] is True  # the job started only after the new folder was saved


def test_a_job_is_refused_when_the_projects_folder_changed_after_the_project_was_opened(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = client.app.state.jobs
    p = ui.projects_dir() / "p"
    p.mkdir(parents=True)
    settings = {"inputs": [str(tmp_path / "clip.mp4")]}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings})
    other = tmp_path / "other"
    other.mkdir()
    real_cmd = ui.cutpoint_cmd

    def cmd_while_the_folder_changes(*args: str) -> list[str]:
        config.update_config(lambda _d: ({"projects_dir": str(other)}, ()))  # a second tab saves
        return real_cmd(*args)

    patch_ui(monkeypatch, "cutpoint_cmd", cmd_while_the_folder_changes)
    try:
        res = client.post("/api/projects/p/jobs", json={"action": "analyze"})
        assert res.status_code == 409 and "projects folder changed" in res.json()["detail"]
        assert not jobs.jobs
    finally:
        for job in list(jobs.jobs.values()):
            jobs.cancel(job.id)


def project_with_link(
    client: TestClient, tmp_path: Path, what: str, data: object = None
) -> tuple[str, Path, Path]:
    """A project whose ``what`` (``ui`` or a metadata file) is a link into a folder outside it.

    Returns (project id, project folder, the link's target); the target holds ``data`` as JSON.
    """
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / Path(what).name
    if what == "ui":
        shutil.move(p / "ui", target)
        link_dir(p / "ui", target)
    else:
        cache.save_json(target, data if data is not None else {})
        (p / what).unlink(missing_ok=True)
        link_file(p / what, target)
    return pid, p, target


@pytest.mark.parametrize("what", ["ui", "highlights.json", "no meta", "review", "exports"])
def test_metadata_read_through_a_link_outside_the_project_is_ignored(
    client: TestClient, tmp_path: Path, what: str
) -> None:
    secret = {"segments": [{"id": "h001"}], "candidates": [{"id": "h001"}], "style": "funny"}
    if what in {"ui", "no meta"}:  # a link alone does not make a folder a project
        pid, p, _ = project_with_link(client, tmp_path, "ui" if what == "ui" else "highlights.json")
        if what == "no meta":
            shutil.rmtree(p / "ui")
        assert client.get("/api/projects").json() == {"projects": []}
        assert client.get(f"/api/projects/{pid}").status_code == 404
        return
    if what == "highlights.json":
        pid, _, _ = project_with_link(client, tmp_path, what, secret)
        got = client.get(f"/api/projects/{pid}").json()
        assert got["highlights"] is None
    elif what == "review":
        pid, _, _ = project_with_link(client, tmp_path, "ui/review.json", secret)
        got = client.get(f"/api/projects/{pid}").json()
        assert got["review"] is None
    else:
        pid, p, target = project_with_link(client, tmp_path, "ui/exports.json")
        (p / "clips_ui_1").mkdir()
        cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": [{"label": "x"}]})
        (p / "final.mp4").write_bytes(b"x")
        cache.save_json(target, {"exports": [{"dir": "clips_ui_1", "final": "final.mp4"}]})
        got = client.get(f"/api/projects/{pid}").json()
        assert got["clips"] is None
        assert [o["name"] for o in got["outputs"]] == ["final.mp4"]  # listed by its name alone
        assert got["final_current"] is True  # no usable export record: nothing is pending


@pytest.mark.parametrize("what", ["review", "clips", "analyze", "looping review"])
def test_metadata_written_through_a_link_outside_the_project_is_refused(
    client: TestClient, tmp_path: Path, what: str
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    src = str((folder / "a.mp4").resolve())
    seg = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    cache.save_json(p / "highlights.json", {"sources": [{"path": src, "duration": 10.0}]})
    cache.save_json(p / "ui" / "review.json", {"segments": [seg]})
    outside = tmp_path / "outside"
    outside.mkdir()
    link = {
        "review": "ui/review.json",
        "clips": "ui/exports.json",
        "analyze": "highlights.json",
        "looping review": "ui/review.json",
    }[what]
    target = outside / "target.json"
    cache.save_json(target, {"keep": True})
    (p / link).unlink(missing_ok=True)
    if what == "looping review":
        link_file(p / link, p / link)  # a link to itself: resolving it fails
    else:
        link_file(p / link, target)
    before = target.read_bytes()
    base = f"/api/projects/{pid}"

    if what in {"review", "looping review"}:
        res = client.put(f"{base}/review", json={"segments": [seg]})
    else:
        res = client.post(f"{base}/jobs", json={"action": what})
    assert res.status_code == 409, res.text
    assert client.get("/api/jobs").json() == {"jobs": []}
    assert target.read_bytes() == before
    assert sorted(f.name for f in outside.iterdir()) == ["target.json"]
    assert not list((p / "ui").glob("segments-*.json"))  # no half-made export either


@pytest.fixture
def hevc_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, Path, Path, Path]:
    """A project with an HEVC source (a browser cannot play it) and an empty ``outside`` folder.

    Returns (project id, source folder, project folder, the outside folder).
    """
    probed = media_info(vcodec="hevc", duration=10.0)
    patch_ui(monkeypatch, "probe", lambda _path: probed)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    outside = tmp_path / "outside"
    outside.mkdir()
    return pid, folder, ui.project_dir(pid), outside


def media_link(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path], sub: str
) -> tuple[str, str, Path, Path, str]:
    """A project with a source the browser cannot play, ``ui/<sub>`` linked to a folder outside.

    Returns (project id, source id, project folder, the outside folder, the source's key).
    """
    pid, folder, p, outside = hevc_project
    (p / "ui").mkdir(exist_ok=True)
    link_dir(p / "ui" / sub, outside)
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    return pid, sid, p, outside, ui.Media.key((folder / "a.mp4").resolve()) or ""


def test_proxy_in_a_cache_folder_linked_outside_is_not_served_or_built(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path]
) -> None:
    pid, sid, _, outside, key = media_link(client, hevc_project, "proxy")
    (outside / f"{key}.mp4").write_bytes(b"outside")
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert client.post(f"/api/projects/{pid}/sources/{sid}/prepare").status_code == 409
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert src["status"] == "needs-proxy"  # probed by the prepare request
    assert client.get(src["url"]).status_code == 409
    assert [f.name for f in outside.iterdir()] == [f"{key}.mp4"]


def test_frames_folder_linked_outside_is_not_used_and_nothing_there_is_deleted(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, sid, _, outside, key = media_link(client, hevc_project, "frames")
    stale = [outside / f"{'0' * 16}-20-160.jpg", outside / f"{key}-30-160.123.part"]
    for f in stale:
        f.write_bytes(b"x")

    def never(*_a: object, **_k: object) -> list[bytes]:
        raise AssertionError("a still must not be built into a linked folder")

    patch_ui(monkeypatch, "grab_images", never)
    token = client.headers["X-Cutpoint-Token"]
    res = client.get(f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={token}")
    assert res.status_code == 409
    assert sorted(f.name for f in outside.iterdir()) == sorted(f.name for f in stale)


@pytest.mark.parametrize("refused", [False, True], ids=["crash", "refused-write"])
def test_a_failing_post_processing_step_fails_the_job_with_a_clean_message(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    def on_success() -> None:
        raise ui.bad("x", 409) if refused else RuntimeError("boom")

    FakeProc().fake_spawn(monkeypatch)
    jobs = ui.Jobs()
    with caplog.at_level(logging.ERROR):
        job = jobs.start("p", "clips", ["x"], on_success=on_success)
        assert job.done.wait(10)
    assert (job.status, job.error) == ("failed", "x" if refused else "boom")
    # a refusal is expected and plain; only a real crash logs its traceback
    assert any(r.exc_info for r in caplog.records if r.levelno >= logging.ERROR) is not refused


def cli_project(
    tmp_path: Path,
    *,
    settings: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    **hl_over: object,
) -> tuple[Path, dict[str, object]]:
    """The command-line project ``cli``: ``highlights.json`` only, no ``ui`` folder. Returns it
    and a segment. ``settings``: extra ``settings`` it records; ``hl_over``: extra top-level
    ``highlights.json`` keys (such as ``style``); ``meta``: written as ``ui/meta.json``."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    p = ui.projects_dir() / "cli"
    p.mkdir(parents=True)
    hl = {
        "sources": [{"path": str(src), "duration": 10.0}],
        "settings": {"inputs": [str(src)], **(settings or {})},
        **hl_over,
    }
    cache.save_json(p / "highlights.json", hl)
    if meta is not None:
        write(p / "ui" / "meta.json", json.dumps(meta))
    seg = {"id": "h001", "source": str(src), "start": 1, "end": 4.5, "label": "good"}
    return p, seg


def test_a_command_line_project_without_a_ui_folder_can_be_reviewed_and_exported(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    assert not (p / "ui").exists()
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    assert client.get("/api/projects/cli").json()["review"]["segments"][0]["id"] == "h001"

    FakeProc().fake_spawn(monkeypatch)
    res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)
    assert len(list((p / "ui").glob("segments-*.json"))) == 1


def test_review_through_a_dangling_ui_link_is_refused_and_creates_nothing_outside(
    client: TestClient, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    outside = tmp_path / "outside" / "ui"
    outside.parent.mkdir()
    link_dir(p / "ui", outside)  # the target does not exist
    res = client.put("/api/projects/cli/review", json={"segments": [seg]})
    assert res.status_code == 409, res.text
    assert not outside.exists() and list(outside.parent.iterdir()) == []


def test_a_link_in_the_projects_folder_to_an_outside_analysis_is_not_a_project(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    cache.save_json(outside / "highlights.json", {"candidates": []})
    link_dir(ui.projects_dir() / "linked", outside)
    listed = [x["id"] for x in client.get("/api/projects").json()["projects"]]
    assert listed == [pid]
    assert client.get("/api/projects/linked").status_code == 404


def swap_ui(p: Path) -> None:
    """Replace ``p/ui`` with a link to a folder that does not exist (inside p)."""
    shutil.rmtree(p / "ui", ignore_errors=True)
    link_dir(p / "ui", p / "gone")


@pytest.mark.parametrize("via", ["review", "clips job"])
def test_a_dangling_ui_link_is_a_409_and_nothing_is_created(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, via: str
) -> None:
    p, seg = cli_project(tmp_path)
    if via == "review":
        swap_ui(p)
        res = client.put("/api/projects/cli/review", json={"segments": [seg]})
    else:  # the review was read, then ui was replaced: the job must not write through it
        assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200

        def cmd_after_ui_changes(*_a: str) -> list[str]:
            swap_ui(p)
            return [sys.executable, "-c", "pass"]

        patch_ui(monkeypatch, "cutpoint_cmd", cmd_after_ui_changes)
        res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 409, res.text
    assert res.json()["detail"] == "cli/ui is not a folder; remove or rename it"
    assert not client.app.state.jobs.jobs
    assert sorted(f.name for f in p.rglob("*") if f.name != "ui") == ["highlights.json"]
    assert not (p / "gone").exists()


@pytest.mark.slow
@pytest.mark.parametrize("why", ["folder changed", "job running", "cannot start"])
def test_a_clips_job_that_does_not_start_leaves_no_segments_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, why: str
) -> None:
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    jobs = client.app.state.jobs
    busy = None
    if why == "folder changed":
        real_cmd = ui.cutpoint_cmd

        def cmd_while_the_folder_changes(*args: str) -> list[str]:
            config.update_config(lambda _d: ({"projects_dir": str(tmp_path / "other")}, ()))
            return real_cmd(*args)

        patch_ui(monkeypatch, "cutpoint_cmd", cmd_while_the_folder_changes)
    elif why == "job running":
        busy = jobs.start("x", "analyze", [sys.executable, "-c", "import time; time.sleep(60)"])
    else:

        def no_spawn(_args: list[str], **_kw: object) -> None:
            raise OSError("not found")

        patch_ui(monkeypatch, "spawn", no_spawn)
    try:
        res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
        assert res.status_code == (500 if why == "cannot start" else 409), res.text
        assert not list(p.rglob("segments-*.json"))
    finally:
        if busy:
            jobs.cancel(busy.id)


def test_one_unreadable_entry_does_not_fail_the_project_list(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("bad", "good"):
        (ui.projects_dir() / name).mkdir(parents=True)
        cache.save_json(ui.projects_dir() / name / "highlights.json", {"candidates": []})
    real_is_dir = Path.is_dir

    def is_dir(self: Path) -> bool:
        if self.name == "bad":
            raise PermissionError("denied")
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    res = client.get("/api/projects")
    assert res.status_code == 200
    assert [x["id"] for x in res.json()["projects"]] == ["good"]


def test_a_proxy_linked_during_the_encode_fails_with_a_warning_and_leaves_nothing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _pid, folder, p, outside = hevc_project
    patch_ui(monkeypatch, "proxy_args", lambda _real, tmp, _n, **_kw: [str(tmp)])

    class Encoder:
        returncode = 0

        def __init__(self, args: list[str], **_kw: object) -> None:
            self.tmp = Path(next(a for a in args if a.endswith(".part.mp4")))

        stdout: ClassVar[list[str]] = []

        def wait(self) -> int:
            shutil.rmtree(p / "ui" / "proxy")  # ui/proxy becomes a link while the encode runs
            link_dir(p / "ui" / "proxy", outside)
            self.tmp.write_bytes(b"proxy")
            return self.returncode

    patch_ui(monkeypatch, "spawn", Encoder)
    with caplog.at_level(logging.INFO):
        media = client.app.state.media
        media.prepare(p, str(folder / "a.mp4"))
        media.pool.submit(int).result(30)  # the one worker has finished the build
    assert list(outside.iterdir()) == []
    assert not list(p.rglob("*.part.mp4"))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [r.levelno for r in warnings] == [logging.WARNING] and not warnings[0].exc_info
    assert "resolves outside the project folder" in warnings[0].getMessage()
    assert media.status(p, str(folder / "a.mp4"))[0] == "failed"


def test_assemble_picks_a_fresh_name_when_only_a_caption_file_is_left(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover ``final_ui_<stamp>.srt`` (no video) is a taken name: nothing may overwrite it."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / "001_a.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    (p / "final_ui_1.srt").write_text("keep me", encoding="utf-8")
    seen: list[list[str]] = []
    proc = FakeProc()
    proc.release.set()
    patch_ui(monkeypatch, "spawn", lambda args, **_kw: seen.append(args) or proc)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)
    assert seen[0][seen[0].index("--out") + 1] == str(p / "final_ui_1-2.mp4")
    assert (p / "final_ui_1.srt").read_text(encoding="utf-8") == "keep me"


class FakeProc:
    """A job process that stays alive until ``release`` is set, then prints ``lines``, exits ``code``."""

    def __init__(self, lines: list[str] | None = None, code: int = 0) -> None:
        self.release = threading.Event()
        self.stdout = self
        self.lines = [f"{ln}\n" for ln in lines or []]
        self.code = code

    def __iter__(self):
        self.release.wait(10)
        return iter(self.lines)

    def wait(self, timeout: float | None = None) -> int:
        return self.code

    def fake_spawn(self, monkeypatch: pytest.MonkeyPatch) -> FakeProc:
        """Make jobs start this process (already exited) instead of a real interpreter.

        ``ui.spawn`` runs a real child at below-normal priority, which is slow on a busy machine.
        """
        self.release.set()
        patch_ui(monkeypatch, "spawn", lambda args, **_kw: self)
        return self


def test_a_cancelled_job_reads_cancelling_until_the_process_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc()
    killed = threading.Event()
    patch_ui(monkeypatch, "spawn", lambda args: proc)
    monkeypatch.setattr(procs, "kill_tree", lambda _p: killed.set())  # the exit comes later
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", ["x"])
    assert job.view()["status"] == "running"
    canceller = threading.Thread(target=jobs.cancel, args=(job.id,), daemon=True)
    canceller.start()
    try:
        assert killed.wait(10)
        assert job.view()["status"] == "cancelling"
        with pytest.raises(HTTPException) as busy:  # it still holds the server
            jobs.start("p", "clips", ["x"])
        assert busy.value.status_code == 409
    finally:
        proc.release.set()
        canceller.join(10)
    assert job.done.wait(10)
    deadline = time.time() + 10
    while job.view()["status"] != "cancelled" and time.time() < deadline:
        time.sleep(0.01)
    assert job.view()["status"] == "cancelled"


def test_cancel_returns_before_the_kill_finishes_and_a_second_cancel_adds_no_killer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc()
    started = threading.Event()
    finish = threading.Event()
    calls: list[object] = []

    def slow_kill(p: object, *a: object) -> None:
        calls.append(p)
        started.set()
        finish.wait(10)
        proc.release.set()

    patch_ui(monkeypatch, "spawn", lambda args: proc)
    monkeypatch.setattr(procs, "kill_tree", slow_kill)
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", ["x"])
    try:
        t0 = time.monotonic()
        jobs.cancel(job.id)  # must not wait for the killer
        assert time.monotonic() - t0 < 5
        assert started.wait(10)
        assert not finish.is_set()
        assert job.view()["status"] == "cancelling"
        jobs.cancel(job.id)
        jobs.cancel(job.id)
        assert calls == [proc]
    finally:
        finish.set()
        proc.release.set()
    assert job.done.wait(10)
    assert len(calls) == 1


def record_export(p: Path, final: str | None) -> None:
    """An export record for ``p`` with a clips folder, whose final is ``final`` (None: not yet)."""
    (p / "clips_ui_1").mkdir(exist_ok=True)
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1", "final": final}]})


def touch(path: Path, age: int) -> None:
    """Create ``path`` as a file last modified ``age`` seconds ago."""
    path.write_bytes(b"x")
    then = time.time() - age
    os.utime(path, (then, then))


def listed_outputs(client: TestClient, pid: str) -> list[str]:
    return [o["name"] for o in client.get(f"/api/projects/{pid}").json()["outputs"]]


def test_every_final_in_the_project_folder_is_listed_newest_first(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    touch(p / "final.mp4", 300)  # from the command line: no export record
    touch(p / "clip-final.mp4", 100)  # from ``edit``, with timelines
    touch(p / "clip-final.fcpxml", 100)
    touch(p / "clip-final.otio", 100)
    touch(p / "clip-final.srt", 100)
    touch(p / "x-final_vertical.mp4", 200)
    touch(p / "x-final_vertical.otio", 200)
    for hidden in (".final.part.mp4", ".final.stage.mp4"):  # a run in progress or killed
        touch(p / hidden, 1)
    (p / "final_dir.mp4").mkdir()
    outside = tmp_path / "outside.mp4"
    touch(outside, 1)
    link_file(p / "final_out.mp4", outside)
    assert listed_outputs(client, pid) == [
        "clip-final.mp4", "clip-final.fcpxml", "clip-final.otio", "clip-final.srt",
        "x-final_vertical.mp4", "x-final_vertical.otio",
        "final.mp4",
    ]  # fmt: skip
    out = client.get(f"/api/projects/{pid}").json()["outputs"]
    srt = next(o for o in out if o["name"] == "clip-final.srt")
    res = client.get(f"{srt['url']}&download=1")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert "attachment" in res.headers["content-disposition"]


def test_a_command_line_final_can_be_downloaded_and_revealed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    touch(p / "final.mp4", 10)
    out = client.get(f"/api/projects/{pid}").json()["outputs"]
    assert client.get(out[0]["url"]).status_code == 200
    res = client.post(f"/api/projects/{pid}/reveal", params={"target": "outputs"})
    assert res.status_code == 200 and opened == [p]


@pytest.mark.parametrize(
    ("final", "listed", "current"),
    [
        ("none", ["final.mp4", "final_ui_1.mp4"], True),  # no export record
        ("final_ui_1.mp4", ["final_ui_1.mp4", "final.mp4"], True),  # its own, though older
        (None, ["final.mp4", "final_ui_1.mp4"], False),  # not assembled yet: the earlier leads
    ],
    ids=["no-record", "own-final-first", "pending-export"],
)
def test_final_current_says_whether_the_leading_final_is_the_exports_own(
    client: TestClient, tmp_path: Path, final: str | None, listed: list[str], current: bool
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    touch(p / "final_ui_1.mp4", 500)
    touch(p / "final.mp4", 5)
    if final != "none":
        record_export(p, final)
    got = client.get(f"/api/projects/{pid}").json()
    assert [o["name"] for o in got["outputs"]] == listed  # no duplicates either
    assert got["final_current"] is current


def test_new_project_that_cannot_save_its_meta_leaves_no_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #194: the failed request removes the project and ``ui/`` (and the projects folder
    only when this call made it)."""
    root = ui.projects_dir()

    def full_disk(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cache, "save_json", full_disk)
    for pre_existing in (False, True):
        if pre_existing:
            root.mkdir(parents=True)
        with pytest.raises(OSError, match="disk full"):
            ui.new_project("Trip", str(tmp_path), "funny")
        assert root.is_dir() is pre_existing
        if pre_existing:
            assert list(root.iterdir()) == []


def test_failed_proxy_build_on_a_command_line_project_removes_the_ui_folders_it_made(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #194: a CLI project has no ``ui/``; a failed build removes ``ui/proxy`` and ``ui/``
    it created."""
    patch_ui(monkeypatch, "probe", lambda _path: media_info(vcodec="hevc", duration=10.0))
    src = tmp_path / "footage" / "a.mp4"
    src.parent.mkdir()
    src.write_bytes(b"x")
    p = ui.projects_dir() / "cli-proj"
    p.mkdir(parents=True)
    cache.save_json(p / "highlights.json", {"settings": {"inputs": [str(src.parent)]}})

    class Failing:
        returncode = 1

        def __init__(self, _args: list[str], **_kw: object) -> None:
            pass

        stdout: ClassVar[list[str]] = ["encoder failed\n"]

        def wait(self) -> int:
            return self.returncode

    patch_ui(monkeypatch, "spawn", Failing)
    media = client.app.state.media
    media.prepare(p, str(src))
    media.pool.submit(int).result(30)  # the one worker has finished the build
    assert media.status(p, str(src))[0] == "failed"
    assert not (p / "ui").exists()


# --- strain: priority, connection resets, proxy encoder ---


@pytest.mark.slow
def test_spawn_runs_children_below_normal_priority_and_asks_for_progress_lines() -> None:
    """Jobs and proxy encodes (and what they start) yield the CPU to the desktop."""
    if os.name == "nt":
        code = (
            "import ctypes; k = ctypes.windll.kernel32; "
            "k.GetCurrentProcess.restype = ctypes.c_void_p; "
            "k.GetPriorityClass.argtypes = [ctypes.c_void_p]; "
            "import os; print(k.GetPriorityClass(k.GetCurrentProcess()), "
            "os.environ.get('CUTPOINT_PROGRESS'))"
        )
        want = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        code = "import os; print(os.nice(0), os.environ.get('CUTPOINT_PROGRESS'))"
        want = min(os.nice(0) + 10, 19)
    proc = ui.spawn([sys.executable, "-c", code])
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode == 0, out
    priority, progress = out.split()[-2:]
    assert int(priority) == want
    assert progress == "1"  # the CLI is asked for PROGRESS lines


@pytest.mark.skipif(os.name != "nt", reason="Windows creation flags")
def test_spawn_keeps_its_own_process_group_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+Break reaches a job only in its own process group: the priority flag is added to it."""
    seen: dict = {}

    def popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(procs, "join_tree_job", lambda _proc: None)
    ui.spawn(["x"])
    flags = seen["creationflags"]
    assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    assert flags & subprocess.BELOW_NORMAL_PRIORITY_CLASS


@pytest.mark.parametrize("low", [True, False])
def test_spawn_lowers_the_priority_unless_asked_not_to(
    monkeypatch: pytest.MonkeyPatch, low: bool
) -> None:
    """Window proxies (the user waits on them) start at normal priority, still in their own group."""
    seen: dict = {}

    def popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs, args=args)
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(procs, "join_tree_job", lambda _proc: None)
    monkeypatch.setattr(procs.shutil, "which", lambda name: f"/usr/bin/{name}")
    ui.spawn(["ffmpeg"], low_priority=low)
    if os.name == "nt":
        assert seen["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
        assert bool(seen["creationflags"] & subprocess.BELOW_NORMAL_PRIORITY_CLASS) is low
    else:
        assert seen["start_new_session"]
        assert seen["args"] == (["/usr/bin/nice", "-n", "10", "ffmpeg"] if low else ["ffmpeg"])


@pytest.mark.parametrize(
    ("exc", "passed_on"),
    [
        (ConnectionResetError(10054, "reset"), False),
        (ConnectionAbortedError(10053, "aborted"), False),
        (BrokenPipeError(32, "broken pipe"), True),
        (OSError(5, "other"), True),
        (RuntimeError("bug"), True),
        (None, True),
    ],
)
def test_connection_resets_are_not_logged_but_other_loop_errors_are(
    exc: BaseException | None, passed_on: bool
) -> None:
    """A browser cancelling a range request (seek, source switch) is not an error."""
    seen: list[dict] = []
    loop = SimpleNamespace(default_exception_handler=seen.append)
    context = {"message": "Exception in callback", "exception": exc}
    if exc is None:
        del context["exception"]
    ui.quiet_resets(loop, context)  # type: ignore[arg-type]
    assert seen == ([context] if passed_on else [])


def test_the_server_loop_uses_the_quiet_reset_handler() -> None:
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:

        async def handler() -> object:
            return asyncio.get_running_loop().get_exception_handler()

        assert c.portal.call(handler) is ui.quiet_resets


@pytest.mark.parametrize("kind", [None, "nvenc", "vaapi"])
def test_proxy_has_short_keyframe_interval_and_at_most_30_fps(kind: str | None) -> None:
    enc = ui.VideoEncoder(kind, "/dev/dri/renderD128") if kind else None
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, encoder=enc)
    assert args[args.index("-g") + 1] == "30"
    assert args[args.index("-fpsmax") + 1] == "30"
    codec = args[args.index("-c:v") + 1]
    assert codec == {None: "libx264", "nvenc": "h264_nvenc", "vaapi": "h264_vaapi"}[kind]
    assert ("-maxrate" in args) is (kind == "nvenc")  # NVENC's size is capped near libx264's
    # VAAPI frames are uploaded after the scale and never get a -pix_fmt
    assert args[args.index("-vf") + 1].endswith("hwupload") is (kind == "vaapi")
    assert ("-pix_fmt" in args) is (kind != "vaapi")
    assert args[-1] == "out.mp4"


class ProxyEncoder:
    """A fake proxy ffmpeg: records its command; fails when it is NVENC and ``nvenc_fails``."""

    calls: list[list[str]]
    nvenc_fails = False

    def __init__(self, args: list[str], **_kw: object) -> None:
        self.calls.append(args)
        self.returncode = 1 if self.nvenc_fails and "h264_nvenc" in args else 0
        if self.returncode == 0:
            Path(args[-1]).write_bytes(b"proxy")

    @property
    def stdout(self) -> list[str]:
        return [] if self.returncode == 0 else ["nvenc: driver error\n"]

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def proxy_encoder(monkeypatch: pytest.MonkeyPatch) -> type[ProxyEncoder]:
    monkeypatch.setattr(ProxyEncoder, "calls", [], raising=False)
    patch_ui(monkeypatch, "spawn", ProxyEncoder)
    return ProxyEncoder


def build_proxy(client: TestClient, p: Path, src: Path) -> str:
    media = client.app.state.media
    media.prepare(p, str(src))
    media.pool.submit(int).result(30)  # the one worker has finished the build
    return media.status(p, str(src))[0]


@pytest.mark.parametrize(
    ("kind", "codec"),
    [
        (None, "libx264"),  # no GPU encoder works here, or the user chose the CPU
        ("nvenc", "h264_nvenc"),
        ("vaapi", "h264_vaapi"),
        ("amf", "libx264"),  # its proxy size is unmeasured
    ],
)
def test_proxy_uses_the_gpu_encoder_exports_use_else_libx264(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
    kind: str | None,
    codec: str,
) -> None:
    enc = ui.VideoEncoder(kind or "cpu", "/dev/dri/renderD128" if kind == "vaapi" else None)
    patch_ui(monkeypatch, "video_encoder", lambda: enc)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    [args] = proxy_encoder.calls
    assert args[args.index("-c:v") + 1] == codec


def test_a_failed_nvenc_proxy_is_built_again_with_libx264(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The probe encodes 0.2 s of black; a driver can still fail on the real file."""
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("nvenc"))
    monkeypatch.setattr(proxy_encoder, "nvenc_fails", True)
    _, folder, p, _ = hevc_project
    (folder / "b.mp4").write_bytes(b"y")
    with caplog.at_level(logging.WARNING):
        assert build_proxy(client, p, folder / "a.mp4") == "ready"
    codecs = [args[args.index("-c:v") + 1] for args in proxy_encoder.calls]
    assert codecs == ["h264_nvenc", "libx264"]
    assert "retrying with libx264" in caplog.text
    assert not list(p.rglob("*.part.mp4"))
    # libx264 worked where NVENC failed: the next proxy goes straight to libx264
    assert build_proxy(client, p, folder / "b.mp4") == "ready"
    assert [args[args.index("-c:v") + 1] for args in proxy_encoder.calls[2:]] == ["libx264"]


def test_a_proxy_failing_with_nvenc_and_libx264_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("nvenc"))

    class Failing(ProxyEncoder):
        def __init__(self, args: list[str], **_kw: object) -> None:
            self.calls.append(args)
            self.returncode = 1

    patch_ui(monkeypatch, "spawn", Failing)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "failed"
    assert len(proxy_encoder.calls) == 2
    assert not client.app.state.media.gpu_failed  # libx264 did not work either
    assert not list(p.rglob("*.part.mp4"))


def test_no_proxy_ffmpeg_starts_after_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    patch_ui(monkeypatch, "spawn", lambda args, **_kw: spawned.append(args))
    media = ui.Media()
    media.shutdown()
    code, _ = media._encode(["ffmpeg"])  # e.g. the libx264 retry after a failed NVENC encode
    assert code != 0
    assert not spawned


def test_a_proxy_ffmpeg_is_registered_as_it_starts_so_shutdown_kills_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = ui.Media()
    killed: list[object] = []
    monkeypatch.setattr(procs, "kill_tree", lambda proc, wait=0, *, term=True: killed.append(proc))

    class Encode:
        def __init__(self, args: list[str], **_kw: object) -> None:
            assert media.lock.locked()  # started in the same step as the closing check
            self.returncode = 1

        def communicate(self) -> tuple[str, None]:
            assert media.procs == {self}
            media.shutdown()  # the server stops during the encode
            return "", None

    patch_ui(monkeypatch, "spawn", Encode)
    media._encode(["ffmpeg"])
    assert len(killed) == 1
    assert isinstance(killed[0], Encode)
    assert not media.procs


# --- window proxies (#323) ---

HLG = {"color_transfer": "arib-std-b67", "color_primaries": "bt2020", "color_space": "bt2020nc"}
WINDOW_BYTES = b"0123456789"


def test_proxy_args_defaults_are_the_whole_source_command() -> None:
    assert ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1) == [
        *("ffmpeg", "-y", "-nostdin", "-v", "error", "-i", "in.mkv", "-map", "0:v:0"),
        *("-map", "0:a:0?", "-vf", "scale=-2:min(720\\,ih)"),
        *("-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p"),
        *("-g", "30", "-fpsmax", "30", "-c:a", "aac", "-b:a", "128k"),
        *("-movflags", "+faststart", "out.mp4"),
    ]  # fmt: skip


def test_proxy_args_window_seeks_before_the_input_and_limits_after_it() -> None:
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, lo=3.0, hi=33.5)
    assert args[args.index("-ss") + 1] == "3.000"
    assert args.index("-ss") < args.index("-i") < args.index("-t")
    assert args[args.index("-t") + 1] == "30.500"  # the window's length, not hi


def test_proxy_args_color_filters_go_between_scale_and_the_pixel_format_with_bt709_tags() -> None:
    color = "zscale=t=bt709"
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, color=color)
    assert args[args.index("-vf") + 1] == f"scale=-2:min(720\\,ih),{color},format=yuv420p"
    tags = ui.BT709_TAGS
    assert args[args.index(tags[0]) :][: len(tags)] == tags
    plain = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1)
    assert "-colorspace" not in plain


def test_info_has_no_color_filter_for_playable_sources_and_the_sdr_filter_for_hlg_hevc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    media = ui.Media()
    probed = media_info(duration=10.0)
    patch_ui(monkeypatch, "probe", lambda _path: probed)
    assert media.info(src, "k1") == (True, 10.0, 1, "")
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed = hlg
    ok, _, _, color = media.info(src, "k2") or (True, 0, 0, "")
    assert not ok
    assert color == encode.sdr_filter(hlg)
    assert color.startswith("zscale=tin=arib-std-b67")


@pytest.mark.parametrize(
    ("over", "color"),
    [
        ({"width": 640, "height": 480, "color_space": "smpte170m", "color_range": "pc"}, "range"),
        ({"width": 720, "height": 576, "color_range": "pc"}, "range"),  # untagged SD is BT.601
        ({"width": 640, "height": 480, "color_space": "smpte170m", "color_range": "tv"}, ""),
        ({"width": 640, "height": 480}, ""),
        ({"color_space": "bt709", "color_range": "pc"}, ""),  # full-range BT.709: as before
        ({}, ""),  # HD without tags
        ({"pix_fmt": "yuv420p10le", **HLG}, "sdr"),
    ],
)
def test_info_narrows_only_full_range_bt601_proxies_and_keeps_the_others_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, over: dict[str, object], color: str
) -> None:
    """#441: a full-range BT.601 proxy becomes TV range with its BT.601 tags (not BT.709 ones);
    limited BT.601, BT.709 and HDR sources keep their filter, so their proxies keep their key."""
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    src = tmp_path / "a.mkv"
    src.write_bytes(b"x")
    info = media_info(vcodec="mpeg2video", duration=10.0, **over)
    patch_ui(monkeypatch, "probe", lambda _path: info)
    media = ui.Media()
    got = (media.info(src, "k") or (True, 0, 0, ""))[3]
    want = {"range": encode.TV_RANGE, "sdr": encode.sdr_filter(info), "": ""}[color]
    assert got == want == encode.preview_filter(info)
    assert (media._proxy_key("k") == "k") == (color == "")
    args = ui.proxy_args(src, tmp_path / "out.mp4", 1, color=got)
    if color == "range":
        assert args[args.index("-vf") + 1] == (
            "scale=-2:min(720\\,ih),scale=out_range=tv,format=yuv420p"
        )
        assert "-colorspace" not in args  # the BT.601 tags of the frames stay


def test_whole_proxy_of_an_hlg_source_is_converted_to_sdr(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    patch_ui(monkeypatch, "probe", lambda _path: hlg)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    [args] = proxy_encoder.calls
    vf = args[args.index("-vf") + 1]
    assert vf == f"scale=-2:min(720\\,ih),{encode.sdr_filter(hlg)},format=yuv420p"
    assert args[args.index("-colorspace") + 1] == "bt709"


def test_proxy_key_mixes_in_the_colour_filter_only_when_the_source_has_one() -> None:
    media = ui.Media()
    media.probed["sdr"] = (False, 10.0, 1, "")
    media.probed["hdr"] = (False, 10.0, 1, "zscale=a")
    assert media._proxy_key("sdr") == "sdr"  # existing SDR proxies keep their names
    assert media._proxy_key("unprobed") == "unprobed" and media._proxy_key(None) is None
    old = media._proxy_key("hdr")
    assert old and len(old) == 16 and old != "hdr"
    media.probed["hdr"] = (False, 10.0, 1, "zscale=b")  # a changed tone map
    assert media._proxy_key("hdr") not in (old, "hdr")


def test_hlg_proxy_built_with_another_tone_map_is_not_served_but_rebuilt(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    patch_ui(monkeypatch, "probe", lambda _path: hlg)
    _, folder, p, _ = hevc_project
    key = ui.Media.key((folder / "a.mp4").resolve()) or ""
    (p / "ui" / "proxy").mkdir(parents=True)
    stale = p / "ui" / "proxy" / f"{key}.mp4"  # named by the content key only, as before
    stale.write_bytes(b"old tone map")
    assert client.app.state.media.status(p, str(folder / "a.mp4"))[0] == "needs-proxy"
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    assert len(proxy_encoder.calls) == 1
    served = client.app.state.media.status(p, str(folder / "a.mp4"))[1]
    assert served is not None and served != stale and served.name != f"{key}.mp4"


class FakeWindowProc:
    """A fake ffmpeg: a window encode stays alive until finished or killed; a whole one ends."""

    def __init__(self, args: list[str], held: bool, low_priority: bool = True) -> None:
        self.args = args
        self.low_priority = low_priority
        self.returncode: int | None = None
        self.done = threading.Event()
        if not held:
            self.finish(0)

    def finish(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
            if code == 0:
                Path(self.args[-1]).write_bytes(WINDOW_BYTES)
            self.done.set()

    def communicate(self) -> tuple[str, None]:
        self.done.wait(10)
        return "", None

    @property
    def stdout(self) -> list[str]:  # a whole build streams its progress; it ends at once
        return []

    def wait(self) -> int | None:
        self.done.wait(10)
        return self.returncode

    @property
    def start(self) -> float:
        return float(self.args[self.args.index("-ss") + 1])


class FakeWindowFfmpeg:
    """Replaces ``spawn`` and ``kill_tree``; ``hold``: window encodes wait for ``finish``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hold = True
        self.procs: list[FakeWindowProc] = []
        self.killed: list[FakeWindowProc] = []
        self.kill_seen = threading.Event()
        patch_ui(monkeypatch, "spawn", self.spawn)
        monkeypatch.setattr(procs, "kill_tree", self.kill)

    def spawn(self, args: list[str], *, low_priority: bool = True) -> FakeWindowProc:
        proc = FakeWindowProc(args, self.hold and "-ss" in args, low_priority)
        self.procs.append(proc)
        return proc

    def kill(self, proc: FakeWindowProc, wait: float = 0, *, term: bool = True) -> None:
        self.killed.append(proc)
        proc.finish(-9)
        self.kill_seen.set()

    def next_proc(self, n: int) -> FakeWindowProc:
        """The ``n``-th ffmpeg started (0-based), once it has."""
        wait_until(lambda: len(self.procs) > n)
        return self.procs[n]


def wait_until(cond: Callable[[], object], timeout: float = 10, what: str = "") -> None:
    """Poll ``cond`` until it holds; fails after ``timeout`` s, saying ``what`` it waited for."""
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out after {timeout:.0f} s" + (
            f": {what}" if what else ""
        )
        time.sleep(0.01)


def lane_idle(media: ui.Media) -> bool:
    with media.lock:
        return not media.windows_queue and media.window_running is None


@pytest.fixture
def window_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeWindowFfmpeg:
    return FakeWindowFfmpeg(monkeypatch)


@pytest.fixture
def window_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window_ffmpeg: FakeWindowFfmpeg
) -> Iterator[tuple[ui.Media, Path, Path]]:
    """A Media, a project folder and a 1000 s HEVC source; its windows spawn the fake ffmpeg."""
    patch_ui(monkeypatch, "probe", lambda _path: media_info(vcodec="hevc", duration=1000.0))
    media, p, src = ui.Media(), tmp_path / "proj", tmp_path / "a.mp4"
    p.mkdir()
    src.write_bytes(b"x")
    yield media, p, src
    media.shutdown()


def focus(
    media: ui.Media,
    p: Path,
    src: Path,
    lo: float,
    hi: float,
    dur: float | None = None,
    *,
    queue: bool = False,
):
    real = ui.remap_source(str(src))
    key, res = media._lookup(real)
    assert key is not None and res is not None
    if dur is not None:
        res = (res[0], dur, res[2], res[3])
    media._focus(p, real, key, res, lo, hi, queue=queue)
    return key


@pytest.mark.parametrize(
    ("lo", "hi", "dur", "span"),
    [
        (100.0, 110.0, 1000.0, (9000, 13000)),  # 10 s before, 20 s after
        (103.2, 107.9, 1000.0, (9000, 13000)),  # rounded outward to multiples of 5 s
        (2.0, 5.0, 1000.0, (0, 2500)),  # clamped to 0
        (95.0, 99.0, 100.0, (8500, 10000)),  # clamped to the duration
        (100.0, 100.0, 100.0, (9000, 10000)),  # a request at the duration
    ],
)
def test_focus_rounds_the_window_outward_and_clamps_it(
    window_media: tuple[ui.Media, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    lo: float,
    hi: float,
    dur: float,
    span: tuple[int, int],
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = focus(media, p, src, lo, hi, dur)
    wait_until(lambda: lane_idle(media) and media.windows.get((p, key)))
    assert list(media.windows[(p, key)]) == [span]


def test_a_request_at_the_duration_is_covered_by_a_ready_window(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = focus(media, p, src, 95.0, 99.0, 100.0)
    wait_until(lambda: lane_idle(media) and media.windows.get((p, key)))
    focus(media, p, src, 100.0, 100.0, 100.0)  # the end of the source: already inside
    focus(media, p, src, 96.0, 98.0, 100.0)
    assert lane_idle(media)
    assert len(window_ffmpeg.procs) == 1


def test_a_running_window_that_does_not_overlap_the_new_focus_is_killed(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    first = window_ffmpeg.next_proc(0)
    focus(media, p, src, 500, 510)
    assert window_ffmpeg.kill_seen.wait(10)
    assert window_ffmpeg.killed == [first]
    second = window_ffmpeg.next_proc(1)  # the new focus encodes next
    assert second.start == 490.0
    second.finish(0)
    wait_until(lambda: lane_idle(media))
    assert [s for s, _ in media.windows[(p, ui.Media.key(src) or "")]] == [49000]
    assert media.failed_windows(p, str(src)) == []  # killed, not failed


def test_a_running_window_that_overlaps_the_new_focus_is_not_killed(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)  # [90, 130]
    first = window_ffmpeg.next_proc(0)
    focus(media, p, src, 125, 135)  # [115, 155]: overlaps, and past the running one's end
    first.finish(0)
    second = window_ffmpeg.next_proc(1)
    assert second.start == 115.0
    second.finish(0)
    wait_until(lambda: lane_idle(media))
    assert window_ffmpeg.killed == []
    assert len(media.windows[(p, ui.Media.key(src) or "")]) == 2


def test_a_superseded_queued_window_never_spawns_and_neither_does_a_cancelled_one(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    window_ffmpeg.next_proc(0)
    focus(media, p, src, 125, 135)  # queued behind the running one
    focus(media, p, src, 700, 710)  # supersedes the queued one and kills the running one
    window_ffmpeg.next_proc(1).finish(0)
    wait_until(lambda: lane_idle(media))
    assert [pr.start for pr in window_ffmpeg.procs] == [90.0, 690.0]

    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    win.cancelled = True
    assert media._encode(["ffmpeg", "-ss", "0"], win)[0] != 0
    assert len(window_ffmpeg.procs) == 2


def test_a_whole_proxy_build_and_its_progress_belong_to_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each project builds its own proxy file: another project on the same footage does not show
    the first one's encode as its own, and its prepare queues its own build."""
    release = threading.Event()

    class Held:
        returncode = 0

        def __init__(self, args: list[str], **_kw: object) -> None:
            self.out = Path(args[-1])

        @property
        def stdout(self) -> list[str]:
            release.wait(10)
            self.out.write_bytes(b"proxy")
            return []

        def wait(self) -> int:
            return 0

    patch_ui(monkeypatch, "spawn", Held)
    monkeypatch.setattr(ui.Media, "_gpu_encoder", lambda _self: None)
    patch_ui(monkeypatch, "probe", lambda _path: media_info(vcodec="hevc", duration=100.0))
    src, a, b = tmp_path / "a.mp4", tmp_path / "A", tmp_path / "B"
    src.write_bytes(b"x")
    a.mkdir()
    b.mkdir()
    key = ui.Media.key(ui.remap_source(str(src))) or ""
    media = ui.Media()
    try:
        assert media.prepare(a, str(src)) == "preparing"
        wait_until(lambda: (a, key) in media.progress)
        assert media.cached_view(a, str(src)) == ("preparing", True, 0.0)
        assert media.cached_view(b, str(src)) == ("needs-proxy", True, None)
        assert media.prepare(b, str(src)) == "preparing"  # queued behind A's build
        assert media.cached_view(b, str(src)) == ("preparing", True, None)
        release.set()
        wait_until(lambda: media.cached_view(b, str(src))[0] == "ready")
        assert media.cached_view(a, str(src))[0] == "ready"
        wait_until(lambda: media.proxies == {} and media.progress == {})
    finally:
        release.set()
        media.shutdown()


def test_window_encodes_run_at_normal_priority_and_the_whole_proxy_at_low(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    """The whole proxy's encode would starve the window the user waits on: only it yields."""
    media, p, src = window_media
    window_ffmpeg.hold = False  # both encodes end at once
    assert media._encode(["ffmpeg", str(p / "whole.mp4")])[0] == 0
    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    assert media._encode(["ffmpeg", str(p / "window.mp4")], win)[0] == 0
    assert [pr.low_priority for pr in window_ffmpeg.procs] == [True, False]


def test_shutdown_stops_the_window_encode_and_its_lane_and_starts_no_more(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    running = window_ffmpeg.next_proc(0)
    focus(media, p, src, 125, 135)  # queued
    media.shutdown()
    assert window_ffmpeg.killed == [running]
    assert media.window_lane is not None
    media.window_lane.join(10)
    assert not media.window_lane.is_alive()
    focus(media, p, src, 400, 410)
    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    assert media._encode(["ffmpeg", "-ss", "0"], win)[0] != 0
    assert len(window_ffmpeg.procs) == 1
    assert not media.windows_queue
    assert not list(p.rglob("*.part.mp4"))


def test_at_most_eight_windows_are_kept_and_the_least_recently_used_goes_first(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = ""
    for i in range(8):
        key = focus(media, p, src, i * 100 + 20, i * 100 + 25)
        wait_until(lambda: lane_idle(media))
    assert len(media.windows[(p, key)]) == 8
    oldest, second = sorted(media.windows[(p, key)])[:2]
    assert media.window(p, str(src), *oldest) is not None  # used again: now the newest
    focus(media, p, src, 820, 825)
    wait_until(lambda: lane_idle(media))
    kept = media.windows[(p, key)]
    assert len(kept) == 8
    assert oldest in kept and second not in kept
    assert not (p / ui.UI_DIR / "proxy" / f"{key}.w{second[0]}-{second[1]}.mp4").exists()


def age(*files: Path) -> None:
    """Last written or used before prune_proxies' grace period."""
    then = time.time() - ui.PROXY_PRUNE_GRACE_S - 60
    for f in files:
        os.utime(f, (then, then))


def test_windows_are_retired_not_deleted_when_the_whole_proxy_lands_and_pruned_next_run(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = ""
    for lo in (20, 420):
        key = focus(media, p, src, lo, lo + 5)
        wait_until(lambda: lane_idle(media))
    folder = p / ui.UI_DIR / "proxy"
    windows = sorted(folder.glob(f"{key}.w*.mp4"))
    assert len(windows) == 2
    media._build(p, src, folder / f"{key}.mp4", key, 1)
    assert (p, key) not in media.windows
    assert media.windows_of(p, str(src)) == ([], [])  # no longer listed
    assert sorted(folder.glob(f"{key}.w*.mp4")) == windows  # but a page may still play one
    # w1000-4500: 10 s before, 20 s after, from 20 s
    assert media.window(p, str(src), 1000, 4500) == folder / f"{key}.w1000-4500.mp4"
    media.shutdown()

    age(*windows)
    fresh = ui.Media()  # the next run
    fresh.prune_proxies(p, [str(src)])
    assert [f.name for f in folder.iterdir()] == [f"{key}.mp4"]


@pytest.mark.parametrize(
    ("want", "dur_cs", "covered"),
    [
        ((1000, 2000), 10000, True),
        ((999, 2000), 10000, False),  # starts before the window
        ((1000, 2995), 10000, True),  # hi - 5 cs: inside
        ((1000, 2996), 10000, False),  # the window's last frame (4 cs) is not played
        ((1000, 3000), 10000, False),
        ((1000, 3000), 3000, True),  # the window ends with the source: all of it plays
        ((1000, 3001), 3000, False),
        ((1000, 3000), 0, False),  # duration unknown
    ],
)
def test_a_window_covers_a_request_except_its_last_frame_unless_it_ends_the_source(
    want: tuple[int, int], dur_cs: int, covered: bool
) -> None:
    assert ui._covers((1000, 3000), want, dur_cs) is covered


def test_a_request_in_the_last_frame_of_a_ready_window_queues_a_new_one(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = focus(media, p, src, 100, 110)  # [90, 130]
    wait_until(lambda: lane_idle(media) and media.windows.get((p, key)))
    focus(media, p, src, 119.9, 129.95)  # inside: nothing new
    assert len(window_ffmpeg.procs) == 1
    focus(media, p, src, 120.0, 129.97)  # reaches the last frame: a new window
    wait_until(lambda: len(window_ffmpeg.procs) == 2 and lane_idle(media))
    assert window_ffmpeg.procs[1].start == 110.0


def test_a_cancelled_running_encode_does_not_count_as_covering_the_focus(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    first = window_ffmpeg.next_proc(0)
    with media.lock:
        assert media.window_running is not None
        media.window_running.cancelled = True  # e.g. its source's whole proxy landed
    focus(media, p, src, 102, 108)  # inside the running window's range
    first.finish(0)
    second = window_ffmpeg.next_proc(1)
    assert second.start == 90.0
    second.finish(0)
    wait_until(lambda: lane_idle(media))
    assert len(media.windows[(p, ui.Media.key(src) or "")]) == 1


def test_a_window_encode_for_a_source_whose_whole_proxy_lands_is_stopped(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    key = focus(media, p, src, 100, 110)
    running = window_ffmpeg.next_proc(0)
    media._drop_windows(p, key)
    assert window_ffmpeg.kill_seen.wait(10)
    assert window_ffmpeg.killed == [running]
    wait_until(lambda: lane_idle(media))
    assert not list(p.rglob("*.mp4"))
    assert media.failed_windows(p, str(src)) == []  # stopped, not failed


def prepare(client: TestClient, pid: str, sid: str, body: object = None) -> object:
    url = f"/api/projects/{pid}/sources/{sid}/prepare"
    return client.post(url) if body is None else client.post(url, json=body)


@pytest.fixture
def whole_builds(client: TestClient) -> list[tuple]:
    """The whole-proxy builds queued; none runs."""
    queued: list[tuple] = []
    client.app.state.media.pool.submit = lambda fn, *a: queued.append(a)  # type: ignore[method-assign]
    return queued


@pytest.mark.parametrize("body", [None, {}, {"x": 1}], ids=["no-body", "empty", "unrelated"])
def test_prepare_without_a_window_keeps_the_whole_source_behaviour(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    body: object,
) -> None:
    pid, _, _, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    res = prepare(client, pid, sid, body)
    assert (res.status_code, res.json()) == (200, {"status": "preparing"})
    assert len(whole_builds) == 1
    assert not window_ffmpeg.procs
    assert client.app.state.media.window_lane is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"lo": "1", "hi": 5}',
        b'{"lo": 1, "hi": 1' + b"0" * 400 + b"}",  # too large for a float: no 500
        b'{"lo": 6, "hi": 5}',
        b'{"lo": -1, "hi": 5}',
        b'{"lo": 1}',
    ],
)
def test_prepare_refuses_an_invalid_window_and_queues_nothing(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    raw: bytes,
) -> None:
    pid, _, _, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    url = f"/api/projects/{pid}/sources/{sid}/prepare"
    res = client.post(url, content=raw, headers={"Content-Type": "application/json"})
    assert res.status_code == 400, res.text
    assert not whole_builds
    assert not window_ffmpeg.procs
    assert client.app.state.media.proxies == {}


def test_prepare_with_a_window_queues_it_and_the_whole_proxy_and_the_listing_shows_it(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, _, _, _ = hevc_project
    base = f"/api/projects/{pid}"
    sid = client.get(base).json()["sources"][0]["id"]
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4.5}).json() == {"status": "preparing"}
    assert len(whole_builds) == 1
    proc = window_ffmpeg.next_proc(0)
    assert (proc.start, proc.args[proc.args.index("-t") + 1]) == (0.0, "10.000")  # clamped to 10 s

    def probing(_path: object) -> None:
        raise AssertionError("listing a project must not run ffprobe")

    patch_ui(monkeypatch, "probe", probing)
    src = client.get(base).json()["sources"][0]
    assert src["windows"] == []
    assert src["pending"] == [{"lo": 0.0, "hi": 10.0, "state": "encoding"}]
    assert src["duration"] == 10.0  # the page needs it to play the window

    proc.finish(0)
    wait_until(lambda: lane_idle(client.app.state.media))
    src = client.get(base).json()["sources"][0]
    assert src["pending"] == []
    [win] = src["windows"]
    assert (win["lo"], win["hi"]) == (0.0, 10.0)
    assert win["url"].split("?")[0].endswith(f"/sources/{sid}/window/0-1000")


def test_the_listing_shows_a_second_overlapping_window_as_queued(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, folder, p, _ = hevc_project
    media = client.app.state.media
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    real = ui.remap_source(str(folder / "a.mp4"))
    key, res = media._lookup(real)
    res = (res[0], 1000.0, res[2], res[3])
    media._focus(p, real, key, res, 100, 110)
    window_ffmpeg.next_proc(0)
    media._focus(p, real, key, res, 125, 135)
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert [(w["lo"], w["hi"], w["state"]) for w in src["pending"]] == [
        (90.0, 130.0, "encoding"),
        (115.0, 155.0, "queued"),
    ]
    assert src["id"] == sid


# --- whole-build progress and queued windows (#323, part 2) ---


class LineProc:
    """A fake ffmpeg whose merged output is ``lines``; ``seen`` is what ``media.progress[key]``
    held after each line was read (and once more after the last)."""

    def __init__(
        self, media: ui.Media, key: tuple[Path, str], lines: list[str], returncode: int = 0
    ) -> None:
        self.returncode = returncode
        self.seen: list[float | None] = []
        self._media, self._key, self._lines = media, key, lines

    @property
    def stdout(self) -> Iterator[str]:
        for line in self._lines:
            yield line + "\n"
            self.seen.append(self._media.progress.get(self._key))
        self.seen.append(self._media.progress.get(self._key))

    def wait(self) -> int:
        return self.returncode


@pytest.mark.parametrize(
    ("lines", "duration", "expected"),
    [
        (["out_time_us=2500000", "out_time_us=5000000"], 10.0, [0.25, 0.5, 0.5]),
        (["out_time_us=15000000"], 10.0, [1.0, 1.0]),  # a rounding overshoot is capped
        (["out_time_us=N/A", "out_time_us=5000000"], 10.0, [0.0, 0.5, 0.5]),
        (["out_time_us=5000000"], 0.0, [0.0, 0.0]),  # an unknown duration shows nothing
        (
            ["out_time_us=5000000", "out_time_us=3000000", "out_time_us=7000000"],
            10.0,
            [0.5, 0.5, 0.7, 0.7],  # a report that steps back never lowers the bar
        ),
    ],
    ids=["updates", "capped", "not-available", "no-duration", "steps-back"],
)
def test_read_progress_turns_out_time_into_a_share_of_the_duration(
    lines: list[str], duration: float, expected: list[float]
) -> None:
    media, pk = ui.Media(), (Path("p"), "k")
    media.progress[pk] = 0.0
    proc = LineProc(media, pk, lines)
    media._read_progress(proc, pk, duration)  # type: ignore[arg-type]
    assert proc.seen == expected


def test_read_progress_keeps_errors_in_the_tail_and_progress_lines_out_of_it() -> None:
    media = ui.Media()
    proc = LineProc(
        media,
        (Path("p"), "k"),
        [
            "frame=10",
            "bitrate=   4.1kbits/s",  # ffmpeg pads some values
            "out_time_us=1000000",
            "out_time=00:00:01.000000",
            "progress=continue",
            "",
            "[hevc @ 0x1] Invalid NAL unit size",
            "Error while decoding stream #0:0",
            "progress=end",
        ],
    )
    tail = media._read_progress(proc, (Path("p"), "k"), 10.0)  # type: ignore[arg-type]
    assert tail == "[hevc @ 0x1] Invalid NAL unit size\nError while decoding stream #0:0"


def test_a_whole_build_asks_ffmpeg_for_progress_on_stdout(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    [args] = proxy_encoder.calls
    assert args[1:4] == ["-progress", "pipe:1", "-nostats"]
    assert args[0] == "ffmpeg"


class ScriptedEncoder:
    """Whole-build fake ffmpegs that report ``reports`` in turn: ``(out_time_us, exit code)``.
    ``started`` holds ``media.progress[key]`` as each one starts to run."""

    media: ui.Media
    reports: list[tuple[int, int]]
    started: list[float | None]
    calls: list[list[str]]

    def __init__(self, args: list[str], **_kw: object) -> None:
        self.calls.append(args)
        self.us, self.returncode = self.reports[len(self.calls) - 1]
        if self.returncode == 0:
            Path(args[-1]).write_bytes(b"proxy")

    @property
    def stdout(self) -> Iterator[str]:
        self.started.append(next(iter(self.media.progress.values()), None))
        yield f"out_time_us={self.us}\n"
        self.started.append(next(iter(self.media.progress.values()), None))  # after the report

    def wait(self) -> int:
        return self.returncode


def scripted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, reports: list[tuple[int, int]]
) -> type[ScriptedEncoder]:
    monkeypatch.setattr(ScriptedEncoder, "media", client.app.state.media, raising=False)
    monkeypatch.setattr(ScriptedEncoder, "reports", reports, raising=False)
    monkeypatch.setattr(ScriptedEncoder, "started", [], raising=False)
    monkeypatch.setattr(ScriptedEncoder, "calls", [], raising=False)
    patch_ui(monkeypatch, "spawn", ScriptedEncoder)
    return ScriptedEncoder


@pytest.mark.parametrize(("code", "status"), [(0, "ready"), (1, "failed")])
def test_the_whole_build_progress_is_reported_then_cleared_when_it_ends(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    code: int,
    status: str,
) -> None:
    enc = scripted(client, monkeypatch, [(5_000_000, code)])  # of a 10 s source
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("cpu"))
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == status
    assert enc.started == [0.0, 0.5]
    assert client.app.state.media.progress == {}


def test_the_nvenc_retry_restarts_the_progress_at_zero(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
) -> None:
    enc = scripted(client, monkeypatch, [(8_000_000, 1), (2_000_000, 0)])
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("nvenc"))
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    assert len(enc.calls) == 2
    assert enc.started == [0.0, 0.8, 0.0, 0.2]  # the libx264 run did not start at 0.8
    assert client.app.state.media.progress == {}


def test_the_listing_shows_proxy_and_progress_only_for_a_source_that_needs_a_proxy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    whole_builds: list[tuple],
) -> None:
    pid, folder, _p, _ = hevc_project
    media = client.app.state.media
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]

    def entry() -> dict:
        return client.get(f"/api/projects/{pid}").json()["sources"][0]

    assert (entry()["proxy"], entry()["progress"]) == (False, None)  # not probed yet
    assert prepare(client, pid, sid).json() == {"status": "preparing"}
    assert (entry()["proxy"], entry()["progress"]) == (True, None)  # queued, not encoding yet
    key = ui.Media.key(ui.remap_source(str(folder / "a.mp4"))) or ""
    media.progress[_p, key] = 0.123456
    assert (entry()["proxy"], entry()["progress"]) == (True, 0.123)
    media.proxies[_p, key] = "failed"
    media.progress.pop((_p, key))
    assert (entry()["proxy"], entry()["progress"]) == (True, None)


def test_the_listing_of_a_source_the_browser_plays_has_no_proxy_and_no_progress(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
) -> None:
    pid, _, _, _ = hevc_project
    patch_ui(monkeypatch, "probe", lambda _path: media_info(duration=10.0))  # H.264
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    assert prepare(client, pid, sid).json() == {"status": "ready"}
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert (src["status"], src["proxy"], src["progress"]) == ("ready", False, None)


def queued_starts(media: ui.Media) -> list[float]:
    with media.lock:
        return [w.lo / 100 for w in media.windows_queue]


def test_queued_windows_wait_in_order_behind_the_running_one_and_kill_nothing(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)  # [90, 130]
    first = window_ffmpeg.next_proc(0)
    for lo in (300, 500, 700):
        focus(media, p, src, lo, lo + 10, queue=True)
    assert queued_starts(media) == [290.0, 490.0, 690.0]
    assert window_ffmpeg.killed == []
    first.finish(0)
    for n, start in enumerate((290.0, 490.0, 690.0), 1):
        proc = window_ffmpeg.next_proc(n)
        assert proc.start == start
        proc.finish(0)
    wait_until(lambda: lane_idle(media))
    assert window_ffmpeg.killed == []
    assert len(media.windows[(p, ui.Media.key(src) or "")]) == 4


def test_a_queued_window_covered_by_a_queued_or_running_one_is_not_queued_again(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)  # running: [90, 130]
    window_ffmpeg.next_proc(0)
    focus(media, p, src, 102, 108, queue=True)  # inside the running one
    assert queued_starts(media) == []
    focus(media, p, src, 300, 310, queue=True)
    focus(media, p, src, 300, 310, queue=True)  # the same again
    focus(media, p, src, 302, 308, queue=True)  # inside the queued one
    assert queued_starts(media) == [290.0]
    assert window_ffmpeg.killed == []


def test_at_most_four_windows_wait_and_the_running_one_is_not_one_of_them(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    assert ui.PROXY_WINDOWS_QUEUED == 4
    focus(media, p, src, 100, 110)
    window_ffmpeg.next_proc(0)
    for lo in (200, 300, 400, 500, 600, 700):
        focus(media, p, src, lo, lo + 10, queue=True)
    assert queued_starts(media) == [190.0, 290.0, 390.0, 490.0]  # the later ones were refused
    assert window_ffmpeg.killed == []


def test_a_default_focus_after_queued_windows_clears_them_and_stops_the_running_encode(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    first = window_ffmpeg.next_proc(0)
    for lo in (300, 500):
        focus(media, p, src, lo, lo + 10, queue=True)
    focus(media, p, src, 800, 810)  # the user moved on
    assert window_ffmpeg.kill_seen.wait(10)
    assert window_ffmpeg.killed == [first]
    window_ffmpeg.next_proc(1).finish(0)  # the 300 and 500 windows were dropped, not run
    wait_until(lambda: lane_idle(media))
    assert [pr.start for pr in window_ffmpeg.procs] == [90.0, 790.0]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"queue": true}',
        b'{"queue": "true", "lo": 1, "hi": 5}',
    ],
)
def test_prepare_refuses_a_bad_queue_request_and_queues_nothing(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    raw: bytes,
) -> None:
    pid, _, _, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    url = f"/api/projects/{pid}/sources/{sid}/prepare"
    res = client.post(url, content=raw, headers={"Content-Type": "application/json"})
    assert res.status_code == 400, res.text
    assert not whole_builds
    assert not window_ffmpeg.procs
    assert client.app.state.media.proxies == {}


def test_prepare_with_queue_waits_behind_the_running_window_instead_of_replacing_it(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, folder, p, _ = hevc_project
    media = client.app.state.media
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    real = ui.remap_source(str(folder / "a.mp4"))
    key, res = media._lookup(real)
    media._focus(p, real, key, (res[0], 1000.0, res[2], res[3]), 100, 110)
    first = window_ffmpeg.next_proc(0)
    body = {"lo": 500, "hi": 510, "queue": True}
    assert prepare(client, pid, sid, body).json() == {"status": "preparing"}
    assert queued_starts(media) == [0.0]  # clamped to the 10 s source: waits, kills nothing
    assert window_ffmpeg.killed == []
    first.finish(0)
    window_ffmpeg.next_proc(1).finish(0)
    wait_until(lambda: lane_idle(media))
    assert window_ffmpeg.killed == []


def test_a_failed_window_logs_ffmpeg_s_last_lines_and_is_listed_until_asked_for_again(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning says why (ffmpeg's last lines); the listing tells the page to stop waiting."""
    pid, _, _, _ = hevc_project
    base = f"/api/projects/{pid}"
    sid = client.get(base).json()["sources"][0]["id"]
    output = [
        "",
        "ffmpeg version x",
        *[f"line {i}" for i in range(1, 6)],
        "  ",
        "Conversion failed!  ",
    ]

    class Failing:
        returncode = 1

        def __init__(self, args: list[str], **_kw: object) -> None:
            self.args = args

        def communicate(self) -> tuple[str, None]:
            return "\n".join(output) + "\n", None

    patch_ui(monkeypatch, "spawn", Failing)
    monkeypatch.setattr(ui.Media, "_gpu_encoder", lambda _self: None)
    entry = lambda: client.get(base).json()["sources"][0]  # noqa: E731
    with caplog.at_level(logging.WARNING, logger="cutpoint.ui"):
        assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
        wait_until(lambda: entry()["failed"])
    assert entry()["failed"] == [{"lo": 0.0, "hi": 10.0, "reason": "Conversion failed!"}]
    assert entry()["pending"] == []
    [warning] = [r.getMessage() for r in caplog.records if "window proxy" in r.getMessage()]
    assert warning.endswith("failed: line 2\nline 3\nline 4\nline 5\nConversion failed!")

    patch_ui(monkeypatch, "spawn", window_ffmpeg.spawn)
    window_ffmpeg.hold = False
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
    assert entry()["failed"] == []  # asked for again
    wait_until(lambda: entry()["windows"])


@pytest.mark.parametrize(
    ("code", "tail", "reason"),
    [
        (1, "Conversion failed!", "Conversion failed!"),
        (1, "", "ffmpeg exited with code 1"),
        (-9, "", "ffmpeg exited with code -9 (0xFFFFFFF7)"),
        (3221225477, "", "ffmpeg exited with code 3221225477 (0xC0000005)"),
    ],
)
def test_a_window_failure_without_output_names_the_exit_code(
    code: int, tail: str, reason: str
) -> None:
    assert ui._exit_reason(code, tail) == reason


def test_window_endpoint_serves_ranges_and_refuses_what_is_not_ready(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, _, proj, _ = hevc_project
    media = client.app.state.media
    base = f"/api/projects/{pid}"
    token = client.headers["X-Cutpoint-Token"]
    sid = client.get(base).json()["sources"][0]["id"]
    ready = f"{base}/sources/{sid}/window/0-1000?t={token}"
    assert client.get(ready).status_code == 409  # not probed or built yet

    window_ffmpeg.hold = False
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
    wait_until(lambda: client.get(base).json()["sources"][0]["windows"])
    res = client.get(ready, headers={"Range": "bytes=2-5"})
    assert (res.status_code, res.content) == (206, b"2345")
    assert res.headers["content-range"] == "bytes 2-5/10"
    assert client.head(ready).status_code == 200
    assert client.get(f"{base}/sources/{sid}/window/0-500?t={token}").status_code == 409
    for bad in ("abc", "0-", "-5", "1-2-3", "xyz-5"):
        assert client.get(f"{base}/sources/{sid}/window/{bad}?t={token}").status_code == 404, bad
    assert client.get(f"{base}/sources/{'0' * 12}/window/0-1000?t={token}").status_code == 404
    assert (
        client.get(
            f"{base}/sources/{sid}/window/0-1000", headers={"X-Cutpoint-Token": ""}
        ).status_code
        == 403
    )

    key = ui.Media.key(ui.remap_source(client.get(base).json()["sources"][0]["path"])) or ""
    media.proxies[proj, key] = "failed"
    assert client.get(f"{base}/sources/{sid}/window/0-500?t={token}").status_code == 404


def test_window_in_a_cache_folder_linked_outside_is_not_served_and_nothing_there_is_deleted(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, sid, p, outside, key = media_link(client, hevc_project, "proxy")
    media = client.app.state.media
    window = outside / f"{key}.w0-1000.mp4"
    window.write_bytes(b"outside")
    media.windows[(p, key)] = {(0, 1000): window}  # as if it had been recorded before the link
    token = client.headers["X-Cutpoint-Token"]
    assert (
        client.get(f"/api/projects/{pid}/sources/{sid}/window/0-1000?t={token}").status_code == 409
    )
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 409
    assert not window_ffmpeg.procs
    assert sorted(f.name for f in outside.iterdir()) == [window.name]


def prune_setup(tmp_path: Path) -> tuple[ui.Media, Path, list[str], str, str]:
    """A project with two sources; returns (media, ui/proxy folder, their paths, keys...)."""
    p = tmp_path / "proj"
    folder = p / ui.UI_DIR / "proxy"
    folder.mkdir(parents=True)
    paths = []
    for name, data in (("a.mp4", b"x"), ("b.mp4", b"xy")):
        (tmp_path / name).write_bytes(data)
        paths.append(str(tmp_path / name))
    keys = [ui.Media.key(ui.remap_source(s)) or "" for s in paths]
    return ui.Media(), folder, paths, keys[0], keys[1]


@pytest.mark.ffmpeg  # probes the sources with the real ffprobe
def test_prune_proxies_removes_leftovers_and_adopts_the_windows_that_are_still_wanted(
    tmp_path: Path,
) -> None:
    media, folder, paths, a, b = prune_setup(tmp_path)
    stale = "1" * 16
    keep = [f"{a}.w0-500.mp4", f"{a}.w1000-1500.mp4", "notes.txt", f"{a}.w1-2.txt", f"{b}.mp4"]
    gone = [f"{a}.part.mp4", f"{a}.w0-500.part.mp4", f"{stale}.w0-500.mp4", f"{b}.w0-500.mp4"]
    gone.append(f"{stale}.mp4")  # the whole proxy of a source that left
    for name in keep + gone:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    p = folder.parents[1]
    media.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)
    assert set(media.windows[(p, a)]) == {(0, 500), (1000, 1500)}
    assert (p, b) not in media.windows
    (folder / f"{a}.part.mp4").write_bytes(b"x")  # once per run and project
    media.prune_proxies(p, paths)
    assert (folder / f"{a}.part.mp4").exists()


def test_prune_proxies_keeps_hdr_windows_of_the_current_tone_map_before_any_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, _ = prune_setup(tmp_path)
    hdr = hashlib.sha1(f"{a}|{encode.sdr_filter(hlg)}".encode()).hexdigest()[:16]
    (folder / f"{hdr}.w0-500.mp4").write_bytes(b"x")
    (folder / f"{a}.w0-500.mp4").write_bytes(b"x")  # built with an earlier tone map
    age(*folder.iterdir())
    p = folder.parents[1]
    media.prune_proxies(p, paths)  # a new run: nothing probed yet
    assert sorted(f.name for f in folder.iterdir()) == [f"{hdr}.w0-500.mp4"]
    assert set(media.windows[(p, hdr)]) == {(0, 500)}
    assert probed == ["a.mp4"]  # matched: b.mp4 is not probed


def test_hdr_proxies_of_an_earlier_tone_map_go_once_the_current_proxy_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, b = prune_setup(tmp_path)
    monkeypatch.setattr(media.pool, "submit", lambda *_args: None)  # no build
    stale = folder / f"{a}.mp4"  # named by the content key: no current proxy beside it
    stale.write_bytes(b"x")
    age(stale)
    (folder / f"{b}.mp4").write_bytes(b"x")  # another source's
    p = folder.parents[1]
    media.prune_proxies(p, paths)
    assert probed == []  # nothing to tell it from without probing
    assert media.prepare(p, paths[0]) == "preparing"
    assert probed == ["a.mp4"]
    assert stale.is_file()  # left for now, never served: the proxy key holds the filter
    hdr = media._proxy_key(a)
    assert hdr and hdr != a
    (folder / f"{hdr}.mp4").write_bytes(b"x")  # the build landed
    age(folder / f"{hdr}.mp4")
    later = ui.Media()  # the next run: nothing probed yet
    later.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted([f"{hdr}.mp4", f"{b}.mp4"])


def test_prune_proxies_probes_no_source_whose_proxy_it_knows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    hevc = media_info(vcodec="hevc", duration=10.0)
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hevc)
    media, folder, paths, a, b = prune_setup(tmp_path)
    names = [f"{a}.mp4", f"{b}.w0-500.mp4"]
    for name in names:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    media.prune_proxies(folder.parents[1], paths)
    assert probed == []
    assert sorted(f.name for f in folder.iterdir()) == sorted(names)


def test_prune_proxies_keeps_the_proxies_while_a_source_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name))
    nas = tmp_path / "nas"
    nas.mkdir()
    paths = []
    for name, data in (("a.mp4", b"x"), ("b.mp4", b"xy")):
        (nas / name).write_bytes(data)
        paths.append(str(nas / name))
    a, b = (ui.Media.key(ui.remap_source(s)) or "" for s in paths)
    folder = tmp_path / "proj" / ui.UI_DIR / "proxy"
    folder.mkdir(parents=True)
    # an HDR source's proxies would carry other keys: these may be any source's
    keep = [f"{a}.mp4", f"{b}.w0-500.mp4", f"{'1' * 16}.w0-500.mp4", f"{'2' * 16}.mp4"]
    for name in [*keep, f"{a}.part.mp4"]:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    nas.rename(tmp_path / "nas-offline")  # the drive goes offline
    ui.Media().prune_proxies(folder.parents[1], paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)  # the temporary goes
    assert probed == []


def test_prune_proxies_deletes_proxies_idle_a_month_while_a_source_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    sdr = media_info(duration=10.0)
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or sdr)
    media, folder, paths, a, b = prune_setup(tmp_path)
    paths.append(str(tmp_path / "offline" / "c.mp4"))  # a drive that stays offline
    month = [f"{'1' * 16}.mp4", f"{'2' * 16}.w0-500.mp4"]  # also an HDR window
    recent = [f"{'3' * 16}.mp4", f"{'4' * 16}.w0-500.mp4"]  # may be the offline source's
    for name in [*month, *recent, f"{a}.mp4", f"{b}.w0-500.mp4"]:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    for name in [*month, *recent]:
        then = time.time() - 29 * 86400 - 60
        os.utime(folder / name, (then, then))
    os.utime(folder / f"{a}.mp4", (time.time() - 40 * 86400,) * 2)  # a reachable source's
    everything = sorted(f.name for f in folder.iterdir())
    media.prune_proxies(folder.parents[1], paths)
    assert sorted(f.name for f in folder.iterdir()) == everything
    assert probed == []  # nothing idle a month: nothing to probe for
    for name in month:
        then = time.time() - 30 * 86400 - 60
        os.utime(folder / name, (then, then))
    ui.Media().prune_proxies(folder.parents[1], paths)  # a later run
    assert sorted(f.name for f in folder.iterdir()) == sorted(
        [*recent, f"{a}.mp4", f"{b}.w0-500.mp4"]
    )
    assert probed == ["a.mp4", "b.mp4"]  # neither is HDR: no source names them


@pytest.mark.parametrize("window", [False, True])
def test_prune_proxies_keeps_a_reachable_hdr_sources_proxy_idle_a_month_while_one_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: bool
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, _ = prune_setup(tmp_path)
    paths.append(str(tmp_path / "offline" / "c.mp4"))  # a drive that stays offline
    hdr = hashlib.sha1(f"{a}|{encode.sdr_filter(hlg)}".encode()).hexdigest()[:16]
    name = f"{hdr}.w0-500.mp4" if window else f"{hdr}.mp4"  # a whole proxy retires windows
    (folder / name).write_bytes(b"x")
    then = time.time() - 31 * 86400
    os.utime(folder / name, (then, then))
    media.prune_proxies(folder.parents[1], paths)
    assert [f.name for f in folder.iterdir()] == [name]  # not rebuilt for an hour
    assert probed == ["a.mp4"]  # matched: b.mp4 is not probed


def test_prune_proxies_deletes_stale_whole_proxies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    hevc = media_info(vcodec="hevc", duration=10.0)
    patch_ui(monkeypatch, "probe", lambda path: hlg if Path(path).name == "a.mp4" else hevc)
    media, folder, paths, a, b = prune_setup(tmp_path)
    hdr = hashlib.sha1(f"{a}|{encode.sdr_filter(hlg)}".encode()).hexdigest()[:16]
    (tmp_path / "b.mp4").write_bytes(b"changed")  # b's proxy is of its earlier content
    b_now = ui.Media.key(ui.remap_source(paths[1])) or ""
    keep = [f"{hdr}.mp4", f"{b_now}.mp4"]
    gone = [f"{a}.mp4", f"{'2' * 16}.mp4", f"{b}.mp4"]  # earlier tone maps, b before it changed
    for name in keep + gone:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    media.prune_proxies(folder.parents[1], paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)


@pytest.mark.ffmpeg  # probes the sources with the real ffprobe
def test_prune_proxies_keeps_what_another_run_may_be_encoding_or_serving(
    tmp_path: Path,
) -> None:
    media, folder, paths, a, _ = prune_setup(tmp_path)
    stale = "1" * 16
    p = folder.parents[1]
    names = [f"{a}.part.mp4", f"{a}.w0-500.part.mp4", f"{stale}.mp4", f"{stale}.w0-500.mp4"]
    for name in names:
        (folder / name).write_bytes(b"x")  # just written: another run is encoding them
    used = folder / f"{a}.w1000-1500.mp4"
    used.write_bytes(b"x")
    age(used)
    other = ui.Media()  # another run serves the window
    other.windows[(p, a)] = {(1000, 1500): used}
    assert other.window(p, paths[0], 1000, 1500) == used
    (folder / f"{a}.mp4").write_bytes(b"x")  # its whole proxy landed meanwhile
    media.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted([*names, used.name, f"{a}.mp4"])
    fresh = ui.Media()
    age(*folder.iterdir())  # an hour later, nothing in use
    fresh.prune_proxies(p, paths)
    assert [f.name for f in folder.iterdir()] == [f"{a}.mp4"]


@pytest.mark.ffmpeg  # probes the sources with the real ffprobe
def test_prune_proxies_skips_a_file_it_cannot_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media, folder, paths, _, _ = prune_setup(tmp_path)
    for name in ("1" * 16, "2" * 16):
        (folder / f"{name}.mp4").write_bytes(b"x")
    age(*folder.iterdir())
    unlink = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith("1"):
            raise PermissionError(13, "in use by another process", str(self))  # Windows
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    media.prune_proxies(folder.parents[1], paths)
    assert [f.name for f in folder.iterdir()] == [f"{'1' * 16}.mp4"]


def test_a_served_proxy_is_marked_used(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path]
) -> None:
    pid, folder, p, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    source = folder / "a.mp4"
    age(source)
    key = ui.Media.key(source.resolve()) or ""
    (p / ui.UI_DIR / "proxy").mkdir(parents=True)
    proxy = p / ui.UI_DIR / "proxy" / f"{key}.mp4"
    proxy.write_bytes(b"proxy")
    age(proxy)
    before = source.stat().st_mtime_ns
    url = f"/api/projects/{pid}/sources/{sid}?t={client.headers['X-Cutpoint-Token']}"
    first = client.get(url)
    assert first.status_code == 200
    assert first.headers["etag"] == f'"{key}-5"'  # from its name and size, not its mtime
    touched = proxy.stat().st_mtime_ns
    assert time.time() - touched / 1e9 < 60
    assert source.stat().st_mtime_ns == before  # never the source
    seek = client.get(url, headers={"Range": "bytes=1-2", "If-Range": first.headers["etag"]})
    assert (seek.status_code, seek.content) == (206, b"ro")
    assert proxy.stat().st_mtime_ns == touched  # touched recently: not again


def test_a_window_written_recently_is_not_touched_again_when_served(tmp_path: Path) -> None:
    media, folder, paths, a, _ = prune_setup(tmp_path)
    p = folder.parents[1]
    window = folder / f"{a}.w0-500.mp4"
    window.write_bytes(b"x")
    media.windows[(p, a)] = {(0, 500): window}
    touched = window.stat().st_mtime_ns
    assert media.window(p, paths[0], 0, 500) == window
    assert window.stat().st_mtime_ns == touched  # just written: not touched


def test_prune_proxies_adopts_at_most_eight_windows_keeping_the_newest(tmp_path: Path) -> None:
    media, folder, paths, a, _ = prune_setup(tmp_path)
    for i in range(10):
        f = folder / f"{a}.w{i * 1000}-{i * 1000 + 500}.mp4"
        f.write_bytes(b"x")
        os.utime(f, ns=(10**9 * (i + 1),) * 2)
    media.prune_proxies(folder.parents[1], paths)
    assert sorted(s[0] for s in media.windows[(folder.parents[1], a)]) == list(
        range(2000, 10000, 1000)
    )
    assert len(list(folder.iterdir())) == 8


def test_prune_proxies_deletes_nothing_through_a_link_to_a_folder_outside(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path]
) -> None:
    _, _, p, outside, key = media_link(client, hevc_project, "proxy")
    names = [f"{key}.part.mp4", f"{key}.w0-500.mp4", f"{'1' * 16}.w0-500.mp4"]
    for name in names:
        (outside / name).write_bytes(b"x")
    media = ui.Media()
    media.prune_proxies(p, [str(hevc_project[1] / "a.mp4")])
    assert sorted(f.name for f in outside.iterdir()) == sorted(names)
    assert (p, key) not in media.windows


@pytest.mark.slow
@needs_ffmpeg
def test_a_real_window_proxy_keeps_the_flash_and_the_beep_in_sync_from_a_keyframe(
    tmp_path: Path,
) -> None:
    """A window from lo=3.0 of a source with a flash and a beep at 5.0 s: both land at 2.0 s."""
    src = tmp_path / "src.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "color=c=black:s=160x120:r=25:d=12"),
        *("-f", "lavfi", "-i", "aevalsrc='if(between(t,5,5.1),0.8*sin(2*PI*1000*t),0)':s=48000:d=12"),
        *("-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,5,5.03)'"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "250", "-c:a", "aac", str(src)),
    )  # fmt: skip
    p = tmp_path / "proj"
    p.mkdir()
    key = ui.Media.key(src) or ""
    media = ui.Media()
    try:
        media._build_window(ui.ProxyWindow(p, src, key, 300, 1000, 1, ""))
    finally:
        media.shutdown()
    out = p / ui.UI_DIR / "proxy" / f"{key}.w300-1000.mp4"
    assert out.is_file()
    assert not list(p.rglob("*.part.mp4"))

    def probe_csv(*args: str) -> list[list[str]]:
        text = subprocess.run(
            ["ffprobe", "-v", "error", *args, "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        return [line.split(",") for line in text.split()]

    assert float(probe_csv("-show_entries", "format=duration")[0][0]) == pytest.approx(
        7.0, abs=0.15
    )
    frames = probe_csv("-select_streams", "v", "-show_entries", "frame=key_frame,pts_time")
    assert frames[0][0] == "1"  # the first frame is a keyframe
    pts = [float(f[1]) for f in frames]
    gray = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-an", "-vf", "scale=8:8,format=gray",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    means = np.frombuffer(gray, dtype=np.uint8).reshape(-1, 64).mean(axis=1)
    bright = [i for i, m in enumerate(means) if m > 128]
    assert bright, "the flash is missing"
    assert pts[bright[0]] == pytest.approx(2.0, abs=0.04 + 1e-3)  # within one frame at 25 fps

    start = float(probe_csv("-select_streams", "a", "-show_entries", "stream=start_time")[0][0])
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-vn", "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    samples = np.abs(np.frombuffer(pcm, dtype=np.int16).astype(float))
    onset = int(np.argmax(samples > 0.3 * 32768))
    assert samples[onset] > 0.3 * 32768, "the beep is missing"
    assert start + onset / 48000 == pytest.approx(2.0, abs=0.025)


@pytest.mark.slow
@needs_ffmpeg
def test_a_real_whole_build_reports_progress_up_to_the_end_of_the_source(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=s=160x120:r=25:d=3"),
        *("-f", "lavfi", "-i", "sine=f=440:d=3"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)),
    )  # fmt: skip
    p = tmp_path / "proj"
    p.mkdir()
    key = ui.Media.key(src) or ""

    class Recorder(dict):
        def __init__(self) -> None:
            super().__init__()
            self.values: list[float] = []

        def __setitem__(self, k: str, v: float) -> None:
            self.values.append(v)
            super().__setitem__(k, v)

    media = ui.Media()
    media.progress = Recorder()  # type: ignore[assignment]
    try:
        proxy = p / ui.UI_DIR / "proxy" / f"{key}.mp4"
        proxy.parent.mkdir(parents=True)
        media._build(p, src, proxy, key, 1, "", 3.0)
    finally:
        media.shutdown()
    assert proxy.is_file()
    values = media.progress.values  # type: ignore[attr-defined]
    assert values[0] == 0.0
    assert values == sorted(values)
    assert max(values) == pytest.approx(1.0, abs=0.1)
    assert media.progress == {}


# --- source list cache (#129) ---


@pytest.fixture
def folder_listings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry per ``find_videos`` call made by the UI."""
    calls: list[int] = []
    real = ui.find_videos

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    patch_ui(monkeypatch, "find_videos", counting)
    return calls


def source_names(client: TestClient, pid: str) -> list[str]:
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200, res.text
    return sorted(s["name"] for s in res.json()["sources"])


@needs_ffmpeg
def test_media_requests_list_an_unanalyzed_folder_once(
    client: TestClient, tmp_path: Path, folder_listings: list[int]
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    folder_listings.clear()
    base = f"/api/projects/{pid}/sources/{sid}"
    for _ in range(3):
        client.head(base)
        client.get(f"{base}/frame?at=0")
        client.post(f"/api/projects/{pid}/sources/{sid}/prepare")
    assert folder_listings == []


def test_a_video_added_or_removed_in_the_picked_folder_shows_on_the_next_request(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    assert source_names(client, pid) == ["a.mp4"]
    start = folder.stat().st_mtime_ns
    (folder / "b.mp4").write_bytes(b"y")
    os.utime(folder, ns=(start + 10**9, start + 10**9))
    assert source_names(client, pid) == ["a.mp4", "b.mp4"]
    (folder / "a.mp4").unlink()
    os.utime(folder, ns=(start + 2 * 10**9, start + 2 * 10**9))
    assert source_names(client, pid) == ["b.mp4"]


def test_writing_highlights_rebuilds_the_list_from_the_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x", "b.mp4": b"y"})
    assert source_names(client, pid) == ["a.mp4", "b.mp4"]
    hl = {"sources": [{"path": str(folder / "b.mp4"), "duration": 3.0}], "candidates": []}
    (ui.project_dir(pid) / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    assert source_names(client, pid) == ["b.mp4"]
    cache.save_json(
        ui.project_dir(pid) / "highlights.json",
        {"sources": [{"path": str(folder / "a.mp4"), "duration": 3.0}], "candidates": []},
    )
    assert source_names(client, pid) == ["a.mp4"]


def test_project_reports_the_video_length_and_time_of_the_last_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    """#459: from highlights.json's sources and timings; a recovered part counts what it read,
    and highlights.json without timings has none."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x", "b.mp4": b"y"})
    hl = {
        "sources": [
            {"path": str(folder / "a.mp4"), "duration": 600.0},
            {"path": str(folder / "b.mp4"), "duration": 900.0, "offset": 0, "analyzed_s": 300.5},
        ],
        "timings": {"transcribe_s": 40.0, "total_s": 75.25},
        "candidates": [],
    }
    cache.save_json(ui.project_dir(pid) / "highlights.json", hl)
    analysis = client.get(f"/api/projects/{pid}").json()["analysis"]
    assert analysis == {"video_s": 900.5, "took_s": 75.25}
    cache.save_json(ui.project_dir(pid) / "highlights.json", {**hl, "timings": {}})
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None
    # 360: total_s is the flat analysis alone, so the time is pano_timings.json's total_s
    p = ui.project_dir(pid)
    cache.save_json(p / "ui" / "meta.json", {**ui.read_meta(p), "kind": "360"})
    cache.save_json(p / "highlights.json", hl)
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None
    pano = {"scan_s": 12.5, "analyze_s": 90.0, "analyze_total_s": 300.0, "total_s": 320.5}
    cache.save_json(p / "pano_timings.json", pano)
    analysis = client.get(f"/api/projects/{pid}").json()["analysis"]
    assert analysis == {"video_s": 900.5, "took_s": 320.5}
    # --no-events: total_s without the removed analyze_total_s
    cache.save_json(p / "pano_timings.json", {"analyze_s": 90.0, "total_s": 95.0})
    assert client.get(f"/api/projects/{pid}").json()["analysis"]["took_s"] == 95.0
    # --views-only times the views alone and leaves highlights.json as it was: no time
    cache.save_json(p / "pano_timings.json", {"views": {"033": 4.0}, "total_s": 40.0})
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None
    # a pano_timings.json from before total_s: no time, no crash
    del pano["total_s"]
    cache.save_json(p / "pano_timings.json", pano)
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(45, "45 s"), (59.5, "1 min"), (3570, "1 h"), (8040, "2 h 14 min")],
)
@pytest.mark.slow
def test_span_text_formats_analysis_lengths(util_js, seconds: float, text: str) -> None:
    """#459: seconds under a minute, minutes under an hour, then hours and minutes."""
    assert util_js("return U.spanText(args[0]);", seconds) == text


@pytest.mark.slow
@pytest.mark.parametrize(("at", "n", "row"), [(2, 5, 2), (4, 4, 3), (0, 0, -1)])
def test_focus_goes_to_the_row_in_place_of_one_that_left_the_list(
    util_js, at: int, n: int, row: int
) -> None:
    """Add under "Not in sequence" jumped the Moments column to the top: the added card left the
    list and preserveFocus focused the list's first button. util.js rowInPlace picks the row now at
    the card's index instead (the last one past the end, none in an empty list)."""
    assert util_js("return U.rowInPlace(args[0], args[1]);", at, n) == row
    # preserveFocus's use of it: tests/js/workspace.test.mjs


def test_a_failed_folder_listing_is_not_cached_as_an_empty_list(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    real = ui.find_videos
    calls: list[int] = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError("folder briefly unavailable")
        return real(*args, **kwargs)

    patch_ui(monkeypatch, "find_videos", flaky)
    client.get(f"/api/projects/{pid}")  # the failed listing; the response does not matter
    assert source_names(client, pid) == ["a.mp4"]


def test_the_cache_keeps_at_most_max_projects_and_expires_after_its_ttl(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder_listings: list[int]
) -> None:
    patch_ui(monkeypatch, "SOURCES_CACHE_MAX", 1)
    pid1, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    other = tmp_path / "other"
    other.mkdir()
    (other / "c.mp4").write_bytes(b"z")
    pid2 = client.post("/api/projects", json={"name": "Other", "source": str(other)}).json()["id"]
    source_names(client, pid1)
    source_names(client, pid2)  # evicts pid1
    folder_listings.clear()
    source_names(client, pid2)
    assert len(folder_listings) == 0
    source_names(client, pid1)
    assert len(folder_listings) == 1
    patch_ui(monkeypatch, "SOURCES_CACHE_S", 0)
    source_names(client, pid1)
    assert len(folder_listings) == 2


# --- lazy probe and unreachable picked folder ---


def test_opening_a_project_probes_nothing_until_a_source_is_prepared(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes: list[object] = []

    def counting(path):
        probes.append(path)
        return media_info(duration=10.0)

    patch_ui(monkeypatch, "probe", counting)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    source = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert (probes, source["status"]) == ([], "unknown")
    client.post(f"/api/projects/{pid}/sources/{source['id']}/prepare")
    assert client.get(f"/api/projects/{pid}").json()["sources"][0]["status"] == "ready"
    assert len(probes) == 1


def test_a_picked_folder_that_cannot_be_listed_is_a_field_not_an_error(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    moved = tmp_path / "moved"
    folder.rename(moved)
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200, res.text
    assert str(folder) in res.json()["source_error"]
    assert res.json()["sources"] == []
    moved.rename(folder)
    res = client.get(f"/api/projects/{pid}")
    assert res.json()["source_error"] is None
    assert source_names(client, pid) == ["a.mp4"]


def test_an_unanalyzed_folder_missing_in_place_is_listed_at_its_source_remap_target(
    client: TestClient, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"0123456789"})
    copy = tmp_path / "copy"
    write(copy / "a.mp4", b"0123456789")
    write(copy / "skip_unverified.mp4", b"x")  # source_remap_exclude: would be read at the original
    shutil.rmtree(folder)
    patch_ui(monkeypatch, "probe", lambda _path: media_info(duration=10.0))
    remap(home, monkeypatch, folder, copy, exclude=["*unverified*"])
    data = client.get(f"/api/projects/{pid}").json()
    assert data["source_error"] is None
    assert [s["path"] for s in data["sources"]] == [str(folder / "a.mp4")]
    assert source_names(client, pid) == ["a.mp4"]
    src = data["sources"][0]
    client.post(f"/api/projects/{pid}/sources/{src['id']}/prepare")
    assert client.get(src["url"]).content == b"0123456789"


def test_a_source_remap_change_rebuilds_the_list_at_once_and_other_edits_keep_it(
    client: TestClient, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[int] = []
    real = ui.source_entries

    def counting(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    patch_ui(monkeypatch, "source_entries", counting)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})

    def after(**data) -> int:
        """List sources once under this config.json; how many rebuilds that took."""
        set_config(home, **data)
        builds.clear()
        source_names(client, pid)
        return len(builds)

    assert after() == 1  # first listing
    assert after() == 0  # re-saved without a change
    assert after(source_remap={"/old": "/new"}) == 1  # remap added
    assert after(source_remap={"/old": "/new"}) == 0  # same values re-saved
    assert after(source_remap={"/old": "/new"}, unrelated=1) == 0  # other key changed
    assert after(source_remap={"/old": "/new"}, source_remap_exclude=["*skip*"]) == 1  # exclude
    assert after(source_remap={"/old": "/new"}, source_remap_exclude=["*skip*"]) == 0
    assert after(source_remap={"/old": "/other"}, source_remap_exclude=["*skip*"]) == 1  # retarget


SAMPLE_PROGRESS = [
    'PROGRESS {"step":"transcribe","index":1,"total":7,"label":"Transcription","state":"running","pct":0,"detail":"sample.mp4"}',
    "12:00:01 INFO transcribing sample.mp4",
    'PROGRESS {"step":"transcribe","index":1,"total":7,"label":"Transcription","state":"done","pct":100,"detail":null}',
    'PROGRESS {"step":"reactions","index":2,"total":7,"label":"Reactions","state":"skipped","pct":null,"detail":"events extra not installed"}',
    'PROGRESS {"step":"crashes","index":4,"total":7,"label":"Crashes","state":"skipped","pct":null,"detail":"not action-cam footage"}',
    'PROGRESS {"step":"vision","index":7,"total":7,"label":"Vision rating","state":"skipped","pct":null,"detail":"vision is off (Settings, config.json or --vision)"}',
]  # fmt: skip


def progress_job(lines: list[str], monkeypatch: pytest.MonkeyPatch, code: int = 0) -> ui.Job:
    """A finished job whose (faked) child printed ``lines`` and exited ``code``."""
    FakeProc(lines, code).fake_spawn(monkeypatch)
    job = ui.Jobs().start("p", "analyze", ["x"])
    assert job.done.wait(10)
    return job


def test_progress_lines_become_steps_and_stay_out_of_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = progress_job(SAMPLE_PROGRESS, monkeypatch)
    view = job.view()
    assert view["status"] == "completed"
    assert view["logs"] == ["12:00:01 INFO transcribing sample.mp4"]
    assert view["progress"]["total"] == 7
    steps = view["progress"]["steps"]
    assert [(s["step"], s["state"]) for s in steps] == [
        ("transcribe", "done"),  # its latest state only
        ("reactions", "skipped"),
        ("crashes", "skipped"),
        ("vision", "skipped"),
    ]
    assert steps[0] == {
        "step": "transcribe",
        "index": 1,
        "label": "Transcription",
        "state": "done",
        "pct": 100.0,
        "detail": None,
    }
    assert steps[1]["detail"] == "events extra not installed" and steps[1]["pct"] is None


def test_a_step_sent_again_later_keeps_its_index_place_and_a_log_prefix_is_accepted() -> None:
    job = ui.Job("p", "edit")
    for line in [
        'PROGRESS {"step":"keyframes","index":3,"total":8,"label":"Keyframes","state":"done","pct":100,"detail":null}',
        '12:00:02 INFO PROGRESS {"step":"vision","index":6,"total":8,"label":"Vision rating","state":"done","pct":100,"detail":null}',
        'PROGRESS {"step":"keyframes","index":3,"total":8,"label":"Keyframes","state":"running","pct":40.5,"detail":"events"}\r\n',
    ]:  # fmt: skip
        job.add_output(line)
    view = job.view()
    assert [(s["step"], s["state"], s["pct"]) for s in view["progress"]["steps"]] == [
        ("keyframes", "running", 40.5),
        ("vision", "done", 100.0),
    ]
    assert view["progress"]["total"] == 8
    assert view["logs"] == []


@pytest.mark.parametrize(
    "line",
    [
        "PROGRESS {not json",
        "PROGRESS [1, 2]",
        'PROGRESS {"step":"x","index":1,"state":"exploded"}',
        'PROGRESS {"index":1,"state":"done"}',
        'Saying PROGRESS {"step":"x","index":1,"state":"done"} mid-line',
    ],
)
def test_a_malformed_progress_line_stays_in_the_log_as_text(line: str) -> None:
    job = ui.Job("p", "analyze")
    job.add_output(line + "\n")
    assert job.view()["logs"] == [line]
    assert job.view()["progress"] == {"total": None, "steps": []}


def test_odd_progress_fields_are_tidied() -> None:
    step = ui.parse_progress(
        'PROGRESS {"step":"x","index":2,"total":true,"label":"","state":"running","pct":250,"detail":5}'
    )
    assert step == {
        "step": "x",
        "index": 2,
        "total": None,
        "label": "x",
        "state": "running",
        "pct": 100.0,
        "detail": None,
    }
    nan = ui.parse_progress('PROGRESS {"step":"x","index":2,"state":"running","pct":NaN}')
    assert nan is not None and nan["pct"] is None
    huge = ui.parse_progress(
        'PROGRESS {"step":"x","index":2,"state":"running","pct":1' + "0" * 400 + "}"
    )  # too large for a float: no OverflowError in the job watcher
    assert huge is not None and huge["pct"] is None


@pytest.mark.parametrize("code", [0, 1])
def test_a_step_still_running_when_the_job_fails_reads_interrupted(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    job = progress_job([*SAMPLE_PROGRESS[:1], "12:00:03 ERROR boom"], monkeypatch, code)
    view = job.view()
    assert view["status"] == ("completed" if code == 0 else "failed")
    (step,) = view["progress"]["steps"]
    assert step["state"] == ("running" if code == 0 else "interrupted")
    assert step["pct"] == 0.0 and step["detail"] == "sample.mp4"


def test_a_cancelled_job_interrupts_its_running_step(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakeProc()
    patch_ui(monkeypatch, "spawn", lambda args: proc)
    monkeypatch.setattr(procs, "kill_tree", lambda _p: proc.release.set())
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", ["x"])
    try:
        job.add_output(SAMPLE_PROGRESS[0])
        job.add_output(SAMPLE_PROGRESS[3])
        jobs.cancel(job.id)
    finally:
        proc.release.set()
    assert job.done.wait(10)
    assert job.view()["status"] == "cancelled"
    assert [s["state"] for s in job.view()["progress"]["steps"]] == ["interrupted", "skipped"]


# --- Export options: Smart edges, Audio transition (#462) ---


def _segments_written(p: Path) -> list[dict]:
    (seg_file,) = list(p.rglob("segments-*.json"))
    return json.loads(seg_file.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("body", "smart"),
    [({}, False), ({"speech_aware": True, "smart_tail": True}, True)],
)
def test_clips_job_hands_the_trims_to_the_smart_flags(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, smart: bool
) -> None:
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    # the check runs under the jobs lock, which writes the segments file
    captured = JobSpy(monkeypatch).args
    res = client.post("/api/projects/cli/jobs", json={"action": "clips", **body})
    assert res.status_code == 200, res.text
    args = captured[0]
    assert ("--smart-tail" in args) is smart and ("--speech-aware" in args) is smart
    (seg,) = _segments_written(p)
    if smart:  # the flags decide: no per-segment override
        assert "smart_tail" not in seg and "speech_aware" not in seg
    else:  # today's behaviour: exactly the user's trims
        assert seg["smart_tail"] is False and seg["speech_aware"] is False
    assert (seg["start"], seg["end"]) == (1, 4.5)


@pytest.mark.parametrize(
    ("body", "audio"),
    [
        ({}, "cut"),  # old pages send none: follows the video
        ({"transition": "cut"}, "cut"),
        ({"transition": "fade"}, "crossfade"),
        ({"transition": "fade", "audio_transition": "cut"}, "cut"),
        ({"transition": "fade", "audio_transition": "crossfade"}, "crossfade"),
        ({"transition": "cut", "audio_transition": "cut"}, "cut"),
    ],
)
def test_assemble_job_passes_the_audio_transition(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, audio: str
) -> None:
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": "assemble", **body})
    assert res.status_code == 200, res.text
    args = captured[0]
    assert args.count("--audio-transition") == 1
    assert args[args.index("--audio-transition") + 1] == audio
    assert args[args.index("--transition") + 1] == body.get("transition", "cut")


@pytest.mark.parametrize(
    ("extra", "expect"),
    [
        ({}, {}),  # not sent: the CLI's defaults
        ({"transition": "fade", "audio_transition": "cut", "fade_dur": 0.5, "audio_fade": 0,
          "captions": True, "caption_model": "large-v3"},
         {"--fade-dur": "0.5", "--audio-fade": "0", "--model": "large-v3"}),
    ],
)  # fmt: skip
def test_assemble_job_passes_fade_length_audio_fade_and_captions_model_unchanged(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: dict,
    expect: dict,
) -> None:
    """#485: Fade length, Assembly audio fade and Captions model reach assemble as given."""
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": "assemble", **extra})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    flags = ("--fade-dur", "--audio-fade", "--model")
    assert {f: _after(args, f) for f in flags if f in args} == expect
    parsed = cli.build_parser().parse_args(args)
    assert (parsed.fade_dur, parsed.audio_fade, parsed.model) == (
        extra.get("fade_dur", 0.25),
        extra.get("audio_fade", 0.25),
        extra.get("caption_model"),
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"transition": "cut", "fade_dur": 0.5}, "fade_dur needs the fade video transition"),
        ({"captions": False, "caption_model": "small"}, "caption_model needs captions"),
        ({"captions": True, "caption_model": "--force"}, "caption_model must be one of"),
    ],
)
def test_assemble_job_refuses_fade_and_model_values_the_cli_would_not_get(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, message: str
) -> None:
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": "assemble", **body})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"transition": "cut", "audio_transition": "crossfade"}, "needs the fade"),
        ({"audio_transition": "crossfade"}, "needs the fade"),  # transition defaults to cut
        ({"transition": "fade", "audio_transition": "echo"}, "audio_transition must be"),
        ({"transition": "fade", "audio_transition": None}, "audio_transition must be"),
    ],
)
def test_assemble_job_refuses_an_audio_transition_that_does_not_fit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    message: str,
) -> None:
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": "assemble", **body})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("cfg", "is_set", "shown"),
    [
        (None, False, "off"),
        ("light", True, "light"),
        ("strong", True, "strong"),
        ("off", True, "off"),
        ("loud", False, "off"),  # invalid: not a default the page should start from
        (3, False, "off"),
    ],
)
def test_settings_say_whether_config_json_sets_compress_peaks(
    client: TestClient, home: Path, cfg: object, is_set: bool, shown: str
) -> None:
    """The page starts Compress Audio Peaks on Light unless config.json sets a value."""
    if cfg is not None:
        set_config(home, compress_peaks=cfg)
    st = client.get("/api/settings").json()
    assert (st["compress_peaks_set"], st["compress_peaks"]) == (is_set, shown)


# --- Web contract pins for the Export tab (#462) ---


def _web(name: str) -> str:
    return (ui.WEB_DIR / name).read_text(encoding="utf-8")


def _css_decls(css: str, selector: str) -> dict[str, str]:
    """The declarations of the first rule whose selector list is ``selector`` (whitespace and
    comments ignored), as ``{property: value}``: a reformat or reorder of the rule keeps them."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    want = " ".join(selector.split())
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if " ".join(sel.split()) == want:
            pairs = (d.split(":", 1) for d in body.split(";") if ":" in d)
            return {k.strip(): " ".join(v.split()) for k, v in pairs}
    raise AssertionError(f"no rule {selector!r}")


class _Tags(HTMLParser):
    """index.html's elements by id (their attributes) and each select's option values."""

    def __init__(self, html: str) -> None:
        super().__init__()
        self.by_id: dict[str, dict[str, str | None]] = {}
        self.options: dict[str, list[str]] = {}
        self._select: str | None = None
        self.feed(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if a.get("id"):
            self.by_id[a["id"]] = a
        if tag == "select":
            self._select = a.get("id")
            self.options[self._select or ""] = []
        elif tag == "option" and self._select:
            self.options[self._select].append(a.get("value") or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._select = None


def test_side_tabs_and_panel_headers_stay_put_while_a_panel_scrolls() -> None:
    """Regression: the tab bar and the Sequence and AI Chat headers scrolled away with a long
    panel; they are sticky (layout jsdom cannot see). revealRow's scroll padding is in
    tests/js/graphite.test.mjs."""
    css = _web("app.css")
    assert _css_decls(css, ".side-bar")["position"] == "sticky"
    heads = _css_decls(css, "#panelSequence > .col-head, #panelAssistant > .chat-head")
    assert heads["position"] == "sticky"


def test_the_column_minimums_fit_the_narrowest_three_column_window() -> None:
    """Cross-file contract (#516): app.js's MOM_MIN + SIDE_MIN + STAGE_MIN fit app.css's narrowest
    three-column window, and the stage track keeps STAGE_MIN. The splitters' behaviour is in
    tests/js/workspace.test.mjs."""
    js, css = _web("app.js"), _web("app.css")

    def const(name: str) -> int:
        return int(re.search(rf"\bconst {name}\s*=\s*(\d+)\s*;", js)[1])

    stage = const("STAGE_MIN")
    narrow = int(re.search(r"@media \(max-width:\s*(\d+)px\)\s*\{\s*body", css)[1]) + 1
    assert const("MOM_MIN") + const("SIDE_MIN") + stage <= narrow
    cols = _css_decls(css, "#workspace")["grid-template-columns"]
    assert f"minmax({stage}px, 1fr)" in cols


def test_focus_rings_use_the_amber_text_colour_so_light_theme_keeps_contrast() -> None:
    """Contract pin (#462): --amber is a fill (dark ink on it); focus rings and amber text use
    --amber-text, which Light darkens. test_settings_offer_graphite_light_and_blue_themes... holds
    the contrast ratios."""
    css = _web("app.css")
    rings = re.findall(r"[^{}]*:focus-visible[^{}]*\{[^}]*outline[^}]*\}", css)
    assert rings
    for rule in rings:
        for colour in re.findall(r"outline(?:-color)?:[^;]*var\((--[\w-]+)\)", rule):
            if ".cand-toggle" in rule:  # over a still: --amber on a black halo (3:1 on any still)
                assert colour == "--amber", rule
                continue
            if ".sb-block" in rule:  # inside the current block's amber: the block's text colour
                assert colour == "--scrim-ink", rule
                continue
            assert colour == "--amber-text", rule
    assert re.search(r":focus-visible \{ outline: 2px solid var\(--amber-text\)", css)
    light = re.search(r':root\[data-theme="light"\][^{]*\{(.*?)\}', css, re.S)
    assert light and "--amber-text:" in light[1]


# --- Direct controls for clips and extend options (#462) ---


def options_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    *,
    candidates: int = 3,
) -> tuple[object, list[list[str]], list[Callable[[], None] | None], Path]:
    """Start ``body``'s job on a command-line project with ``candidates`` analysed moments (ids
    h001...), a one-moment review and a vertical export ``clips_ui_1``; the jobs lock check runs,
    so the segments file is written. Returns the response, the argv, the on_success callbacks
    and the project folder."""
    p, seg = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["candidates"] = [
        {"id": f"h{i:03d}", "source": seg["source"], "start": i, "end": i + 2, "score": 50}
        for i in range(1, candidates + 1)
    ]
    cache.save_json(p / "highlights.json", hl)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / "001_a.mp4").write_bytes(b"x")
    cache.save_json(
        p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1", "vertical": True}]}
    )
    spy = JobSpy(monkeypatch)  # the check under the jobs lock writes the segments file
    return client.post("/api/projects/cli/jobs", json=body), spy.args, spy.on_success, p


def _cli_args(args: list[str]) -> list[str]:
    """``args`` without the interpreter prefix, after checking the CLI parses them as given."""
    assert args[:3] == ui.cutpoint_cmd()
    cli.build_parser().parse_args(args[3:])  # exits (SystemExit) on an option it does not know
    return args[3:]


def _after(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1, args
    return args[args.index(flag) + 1]


def test_the_default_page_options_build_the_smart_tail_argv(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The page's defaults (sequence, Speech-aware cuts and Smart tails on, Event tails Auto,
    blank numbers) build the argv and segments file."""
    page = {"action": "clips", "picks": "sequence", "vertical": False, "speech_aware": True,
            "smart_tail": True, "event_tails": "auto", "compress_peaks": "light"}  # fmt: skip
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, page)
    assert res.status_code == 200, res.text
    segments = _segments_written(p)
    # the output and segments names carry a per-second stamp
    args = [re.sub(r"\d{8}-\d{6}", "STAMP", a.replace(str(p), "P")) for a in _cli_args(captured[0])]
    assert args[:3] == ["clips", "P", "--segments"] and args[4] == "--out"
    assert args[6:] == ["--smart-tail", "--speech-aware", "--compress-peaks", "light"]
    assert "smart_tail" not in segments[0] and "speech_aware" not in segments[0]


@pytest.mark.parametrize(("speech", "tail"), [(True, False), (False, True), (False, False)])
def test_clips_job_speech_aware_and_smart_tail_are_separate_choices(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    speech: bool,
    tail: bool,
) -> None:
    body = {"action": "clips", "speech_aware": speech, "smart_tail": tail}
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert ("--speech-aware" in args, "--smart-tail" in args) == (speech, tail)
    assert "--no-speech-aware" not in args and "--no-smart-tail" not in args
    (seg,) = _segments_written(p)
    # an exact trim is the segment's own setting; the flag decides otherwise
    assert seg.get("speech_aware", True) is speech and seg.get("smart_tail", True) is tail


def test_clips_job_passes_the_timing_and_padding_fields_it_was_sent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = {"action": "clips", "speech_aware": True, "smart_tail": True, "event_tails": "on",
            "speech_max_extend": 3, "speech_max_lead": 1.25, "tail_min": 0, "tail_max": 4.5,
            "event_window": 6, "extend_start": 1.5, "extend_end": 0}  # fmt: skip
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    expect = {"--speech-max-extend": "3", "--speech-max-lead": "1.25", "--tail-min": "0",
              "--tail-max": "4.5", "--event-window": "6", "--extend-start": "1.5",
              "--extend-end": "0"}  # fmt: skip
    assert {flag: _after(args, flag) for flag in expect} == expect
    assert "--event-tails" in args and "--extend" not in args


@pytest.mark.parametrize(
    ("extra", "expect"),
    [
        ({}, []),  # Default: the CLI's own crop, size and fade
        ({"crop": "auto", "audio_fade": 0}, ["--crop", "auto", "--audio-fade", "0"]),
        ({"crop": "1920:800:0:140", "size": "1280x720", "audio_fade": 0.5},
         ["--crop", "1920:800:0:140", "--size", "1280x720", "--audio-fade", "0.5"]),
    ],
)  # fmt: skip
def test_clips_job_passes_crop_size_and_audio_fade_unchanged(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: dict,
    expect: list[str],
) -> None:
    """#485: the page's Crop, Output size and Clip audio fade reach clips as given."""
    body = {"action": "clips", "picks": "auto", **extra}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    opts = ("--crop", "--size", "--audio-fade")
    got = [a for i, a in enumerate(args) if a in opts or (i and args[i - 1] in opts)]
    assert got == expect
    parsed = cli.build_parser().parse_args(args)
    assert (parsed.crop, parsed.audio_fade) == (extra.get("crop"), extra.get("audio_fade", 0.25))


@pytest.mark.parametrize(
    ("value", "flag"), [("auto", None), ("on", "--event-tails"), ("off", "--no-event-tails")]
)
def test_clips_job_passes_event_tails_on_or_off_and_nothing_for_auto(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str, flag: str
) -> None:
    body = {"action": "clips", "smart_tail": True, "event_tails": value}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert [a for a in args if "event-tails" in a] == ([flag] if flag else [])


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"picks": "best"}, "picks must be 'sequence' or 'auto'"),
        ({"top": 3}, "need automatic picks"),
        ({"order": "score"}, "need automatic picks"),
        ({"speech_aware": "yes"}, "speech_aware must be true or false"),
        ({"smart_tail": 1}, "smart_tail must be true or false"),
        ({"event_tails": "maybe"}, "event_tails must be"),
        ({"event_tails": ["on"]}, "event_tails must be"),  # not a 500 (unhashable)
        ({"smart_tail": False, "event_tails": "on"}, "event tails need smart tails"),
        ({"speech_max_extend": -1}, "speech_max_extend must be 0-60 seconds"),
        ({"speech_max_lead": 61}, "speech_max_lead must be 0-60 seconds"),
        ({"tail_min": "1"}, "tail_min must be 0-60 seconds"),
        ({"tail_max": True}, "tail_max must be 0-60 seconds"),
        # 0 as the CLI refuses it
        ({"event_window": 0}, "event_window must be above 0 and at most 60 seconds"),
        ({"event_window": -2}, "event_window must be above 0 and at most 60 seconds"),
        ({"extend_start": 31}, "extend_start must be 0-30 seconds"),
        ({"extend_end": [1]}, "extend_end must be 0-30 seconds"),
        ({"crop": "--force"}, "crop must be 'auto', 'none' or W:H:X:Y"),
        ({"size": "1281x720"}, "size must be WxH in even pixels"),
    ],
)
def test_clips_job_refuses_option_values_the_cli_would_not_get(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, message: str
) -> None:
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, {"action": "clips", **extra})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured and not list(p.rglob("segments-*.json"))


@pytest.mark.parametrize(
    ("extra", "selection"),
    [
        ({}, []),  # the style's count and order
        ({"top": 2, "max_total": 60, "order": "score"},
         ["--top", "2", "--max-total", "60", "--order", "score"]),
        ({"ids": ["h003", "h001"], "order": "chronological"},
         ["--ids", "h003", "h001", "--order", "chronological"]),
        ({"max_total": 12.5}, ["--max-total", "12.5"]),
    ],
)  # fmt: skip
def test_automatic_picks_hand_the_selection_to_the_cli(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: dict,
    selection: list[str],
) -> None:
    body = {"action": "clips", "picks": "auto", "speech_aware": False, "smart_tail": True, **extra}
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert args[:2] == ["clips", str(p)] and "--segments" not in args
    assert args[2 : 2 + len(selection)] == selection
    assert _after(args, "--out").startswith(str(p / "clips_ui_"))
    # explicit both ways: the style's defaults must not override what the page showed
    assert "--smart-tail" in args and "--no-speech-aware" in args
    assert not list(p.rglob("segments-*.json"))  # no segments file for automatic picks


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"top": 2, "ids": ["h001"]}, "give top or ids, not both"),
        ({"top": 0}, "top must be a whole number"),
        ({"top": 1.5}, "top must be a whole number"),
        ({"top": True}, "top must be a whole number"),
        ({"top": 501}, "top must be a whole number"),
        ({"ids": ["h009"]}, "ids must be different candidate ids"),  # not in the analysis
        ({"ids": ["h001", "h001"]}, "ids must be different candidate ids"),
        ({"ids": ["--force"]}, "ids must be different candidate ids"),
        ({"ids": "h001"}, "ids must be different candidate ids"),
        ({"ids": []}, "ids must be different candidate ids"),
        ({"ids": [1]}, "ids must be different candidate ids"),
        ({"max_total": 0}, "max_total must be above 0"),
        ({"max_total": -5}, "max_total must be above 0"),
        ({"order": "arc"}, "order must be one of story, score, chronological"),
    ],
)
def test_automatic_picks_refuse_a_bad_selection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, message: str
) -> None:
    body = {"action": "clips", "picks": "auto", **extra}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


def test_automatic_picks_need_an_analysis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = {"action": "clips", "picks": "auto"}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body, candidates=0)
    assert res.status_code == 400, res.text
    assert "analyze the footage before exporting automatic picks" in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("extra", "tail"),
    [
        ({}, ["--no-speech-aware"]),
        ({"extend_start": 1, "extend_end": 2.5, "speech_aware": True, "speech_max_extend": 4},
         ["--start", "1", "--end", "2.5", "--speech-aware", "--speech-max-extend", "4"]),
        ({"extend_end": 3}, ["--end", "3", "--no-speech-aware"]),
    ],
)  # fmt: skip
def test_extend_job_re_exports_the_latest_export_into_a_new_folder(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, tail: list
) -> None:
    res, captured, done, p = options_job(
        client, monkeypatch, tmp_path, {"action": "extend", **extra}
    )
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    out = Path(args[3])
    assert args[:3] == ["extend", str(p / "clips_ui_1"), "--out"] and args[4:] == tail
    assert out.parent == p and out.name.startswith("clips_ui_") and not out.exists()
    # once the CLI wrote the folder, it is the current export (vertical as its source)
    out.mkdir()
    cache.save_json(out / "manifest.json", {"clips": []})
    (out / "001_a.mp4").write_bytes(b"x")
    done[0]()
    exports = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["vertical"], e["final"]) for e in exports[1:]] == [(out.name, True, None)]
    assert client.get("/api/projects/cli").json()["clips"]["dir"] == out.name


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"extend_start": -1}, "extend_start must be 0-30 seconds"),
        ({"extend_end": "2"}, "extend_end must be 0-30 seconds"),
        ({"speech_aware": "on"}, "speech_aware must be true or false"),
        ({"speech_max_lead": 99}, "speech_max_lead must be 0-60 seconds"),
    ],
)
def test_extend_job_refuses_bad_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, message: str
) -> None:
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, {"action": "extend", **extra})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


def test_extend_job_needs_an_export(client: TestClient, tmp_path: Path) -> None:
    cli_project(tmp_path)
    res = client.post("/api/projects/cli/jobs", json={"action": "extend"})
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "export clips before extending them"


def test_the_export_fields_take_the_limits_and_choices_the_server_takes() -> None:
    """Cross-language contract (#462): each Export field's min and max and each select's values
    are those the server validates. What the page sends is in tests/js/export.test.mjs."""
    tags = _Tags(_web("index.html"))

    def limits(cid: str) -> tuple[str | None, str | None]:
        a = tags.by_id[cid]
        return a.get("min"), a.get("max")

    assert limits("optTop") == ("1", str(ui.MAX_SEGMENTS))
    assert limits("optMaxTotal")[1] == f"{ui.MAX_TOTAL_S:g}"
    assert limits("optExtendStart") == limits("optExtendEnd") == ("0", f"{ui.MAX_EXTEND_S:g}")
    for cid in ("optSpeechExtend", "optSpeechLead", "optTailMin", "optTailMax"):
        assert limits(cid) == ("0", f"{ui.MAX_TIMING_S:g}")
    assert limits("optEventWindow")[1] == f"{ui.MAX_TIMING_S:g}"
    for cid in ("optAudioFade", "optFadeDur", "optFinalAudioFade"):
        assert limits(cid)[1] == f"{ui.MAX_FADE_S:g}"
    assert sorted(tags.options["optOrder"]) == sorted(["", *ui.CLIP_ORDERS])
    assert tags.options["optEventTails"] == list(ui.EVENT_TAILS)
    assert tags.options["optCaptionModel"] == ["", *ui.TRANSCRIPT_MODELS]


# --- One-click presets: cutpoint edit --kind (#462) ---


def preset_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    *,
    settings: dict | None = None,
    meta: dict | None = None,
    style: str = "funny",
) -> tuple[object, list[list[str]], list[Callable[[], None] | None], Path]:
    """Start ``body``'s edit job on a command-line project analysed in the ``style`` (extra
    ``settings`` recorded, an optional ``ui/meta.json``). Returns as :func:`options_job`."""
    p, _ = cli_project(tmp_path, settings=settings, meta=meta, style=style)
    spy = JobSpy(monkeypatch)
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", **body})
    return res, spy.args, spy.on_success, p


@pytest.mark.parametrize("kind", list(edit.KINDS))
def test_a_preset_runs_edit_with_its_kind_in_the_projects_own_style(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    res, captured, _, p = preset_job(client, monkeypatch, tmp_path, {"kind": kind})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    src = str(tmp_path / "clip.mp4")
    assert args == ["edit", src, "--kind", kind, "--style", "funny", "--out", str(p),
                    "--compress-peaks", "off", "--no-visual-index"]  # fmt: skip


def test_a_preset_passes_vertical_compress_peaks_and_the_settings_vision(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, vision=True)
    body = {"kind": "montage", "vertical": True, "compress_peaks": "strong"}
    settings = {"vision_top": 12, "ollama_model": "m:1"}  # what the analysis recorded
    res, captured, _, _ = preset_job(client, monkeypatch, tmp_path, body, settings=settings)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert "--vertical" in args and _after(args, "--compress-peaks") == "strong"
    assert args[-4:] == ["--vision", "--no-visual-index", "--vision-top=12", "--ollama-model=m:1"]
    assert "--force" not in args


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "kind must be one of highlights, montage, clips, funny-moments"),
        ({"kind": "trailer"}, "kind must be one of"),
        ({"kind": ["clips"]}, "kind must be one of"),  # not a 500 (unhashable)
        ({"kind": "clips", "vertical": "yes"}, "vertical must be true or false"),
        ({"kind": "clips", "force": 1}, "force must be true or false"),
        ({"kind": "clips", "compress_peaks": "max"}, "compress_peaks must be one of"),
    ],
)
def test_a_preset_refuses_bad_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, message: str
) -> None:
    res, captured, _, _ = preset_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("settings", "meta", "reason"),
    [
        ({"exclude": ["*.lrv"]}, None, "--exclude or --recursive"),
        ({"recursive": True}, None, "--exclude or --recursive"),
        ({}, {"kind": "360", "style": "funny", "source": "x"}, "360 projects"),
        # edit has no --model or --no-scenes: it would transcribe or detect scenes again
        ({"whisper_model": "large-v3", "whisper_device": "cuda"}, None, "large-v3 model"),
        ({"whisper_model": "medium", "whisper_device": "cpu"}, None, "medium model"),
        ({"scenes": False}, None, "--no-scenes"),
        ({"recover": True}, None, "--recover"),
        ({"language": "en"}, None, "--language"),
        ({"max_candidates": 40}, None, "--max-candidates"),
        (
            {"must_include_crash": 0.3},
            {"style": "action-cam", "source": "x"},
            "--must-include-crash",
        ),
    ],
)
def test_presets_are_off_with_the_reason_where_edit_cannot_remake_the_analysis(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: dict,
    meta: dict | None,
    reason: str,
) -> None:
    body = {"kind": "highlights"}
    res, captured, _, _ = preset_job(
        client, monkeypatch, tmp_path, body, settings=settings, meta=meta
    )
    assert res.status_code == 400, res.text
    assert reason in res.json()["detail"] and not captured
    assert reason in client.get("/api/projects/cli").json()["preset_problem"]


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "whisper_model": "small",
            "whisper_device": "cpu",
            "scenes": True,
            "language": None,
            "max_candidates": None,
            "recover": False,
        },
        {"whisper_model": "medium", "whisper_device": "cuda", "exclude": [], "recursive": False},
    ],
)
def test_a_project_edit_can_remake_has_no_preset_problem(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: dict
) -> None:
    res, *_ = preset_job(client, monkeypatch, tmp_path, {"kind": "clips"}, settings=settings)
    assert res.status_code == 200, res.text
    view = client.get("/api/projects/cli").json()
    assert (view["preset_problem"], view["preset_taken"]) == (None, [])


@pytest.mark.parametrize("vertical", [False, True])
def test_a_preset_replaces_an_earlier_runs_clips_only_when_confirmed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vertical: bool
) -> None:
    name = "clips_vertical" if vertical else "clips"
    body = {"kind": "clips", "vertical": vertical}
    res, captured, _, p = preset_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200 and "--force" not in captured[0]  # nothing there yet
    write(p / name / "001_a.mp4")  # what that run exported
    assert client.get("/api/projects/cli").json()["preset_taken"] == [name]
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", **body})
    assert res.status_code == 409, res.text
    assert "confirm to replace" in res.json()["detail"] and len(captured) == 1
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", **body, "force": True})
    assert res.status_code == 200, res.text
    assert "--force" in _cli_args(captured[1])


def test_a_finished_preset_is_the_current_export_with_its_final(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    res, _, done, p = preset_job(client, monkeypatch, tmp_path, {"kind": "montage"})
    assert res.status_code == 200, res.text
    # an earlier run's entry for the same folder is replaced, other exports stay
    cache.save_json(
        p / "ui" / "exports.json",
        {"exports": [{"dir": "clips", "final": "old.mp4"}, {"dir": "clips_ui_1", "final": None}]},
    )
    with pytest.raises(RuntimeError, match="without its clips and final"):
        done[0]()
    write(p / "clips" / "001_a.mp4")
    cache.save_json(p / "clips" / "manifest.json", {"clips": []})
    write(p / "final.mp4")
    done[0]()
    exports = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["final"], e.get("vertical")) for e in exports] == [
        ("clips_ui_1", None, None),
        ("clips", "final.mp4", False),
    ]
    view = client.get("/api/projects/cli").json()
    assert view["clips"]["dir"] == "clips" and view["final_current"] is True


def test_the_preset_select_lists_the_kinds_of_edit() -> None:
    """Cross-language contract (#462): the One-Click Edit select offers edit's kinds. Asking
    before replacing is in tests/js/export.test.mjs."""
    assert _Tags(_web("index.html")).options["optKind"] == list(edit.KINDS)


# --- Web contract pins for the AI Chat tab (#346) ---


def test_the_chat_input_is_limited_as_the_server_limits_a_message() -> None:
    """Cross-language contract (#346): the chat input's maxlength is ai_session.MAX_TEXT. The
    panel's markup is in tests/js/markup.test.mjs."""
    assert _Tags(_web("index.html")).by_id["chatInput"]["maxlength"] == str(ai_session.MAX_TEXT)


def test_the_chat_context_sends_only_what_the_consent_names() -> None:
    """What context.py sends a hosted assistant stays within what the AI Chat consent names
    (analysis results only: no audio, no image parts, file names but not paths). The consent
    texts themselves are checked in tests/js/chat.test.mjs and tests/js/models.test.mjs."""
    from cutpoint.assistant import context  # noqa: PLC0415

    cand = {
        "id": "h1",
        "source": "D:/Videos/clip.mp4",
        "start": 1.0,
        "end": 5.0,
        "score": 80,
        "label": "jump",
        "kind": "crash",
        "transcript": "look at that",
        "signals": {"keywords": ["look"], "events": {"laughter": 0.4}},
    }
    table = context.MomentTable([cand], [])
    # kind, seen, cues, heard (#634): what analysis saw, as labels, scores and signals
    disclosed = {"id", "file", "start", "end", "analyzed", "score", "seq", "label", "said"}
    disclosed |= {"kind", "seen", "cues", "heard", "len"}  # len: end - start (as exported)
    assert set(table.row("h1")) <= disclosed
    assert set(table.row("h1", full=True)) <= disclosed | {"signals"}
    msgs = context.build_messages(table, [], "hi", context.budget(8192))
    assert all(isinstance(m["content"], str) for m in msgs)  # text only: no image parts
    assert "D:/Videos" not in "".join(m["content"] for m in msgs)


def test_model_text_is_never_set_as_markup() -> None:
    """Security contract (#346): replies, summaries, labels and notes come from the model or the
    project; no web module has an HTML sink, so they can only be set as text. markdown() is in
    tests/js/markdown.test.mjs."""
    sinks = ("innerHTML", "outerHTML", "insertAdjacentHTML", "DOMParser", "document.write",
             "srcdoc", "createContextualFragment", "setHTMLUnsafe")  # fmt: skip
    inline = re.findall(r"<script\b[^>]*>(.*?)</script>", _web("index.html"), re.S)
    assert any(s.strip() for s in inline)  # the early theme script
    scripts = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(ui.WEB_DIR.glob("*.js"))]
    assert {"app.js", "util.js", "api.js", "picker.js"} <= {name for name, _ in scripts}
    for name, js in [*scripts, *(("index.html <script>", s) for s in inline)]:
        for sink in sinks:
            assert sink not in js, (name, sink)


# --- Moments list order (#469) ---


def _sort_positions(util_js, moments: list[dict]) -> list[str]:
    """Run util.js's positionCompare under node on ``moments``; return their ids in sorted order."""
    return util_js("return args[0].sort(U.positionCompare).map((c) => c.id);", moments)


def _m(id_: str, source: str, start: float) -> dict:
    return {"id": id_, "source": source, "start": start}


@pytest.mark.slow
def test_position_order_is_natural_by_file_then_start_time(util_js) -> None:
    """#469: video order groups moments by source (clip2 before clip10, A-Z), then start time."""
    moments = [
        _m("c10b", "/v/clip10.mp4", 5),
        _m("c2late", "/v/clip2.mp4", 90),
        _m("a", "/v/Alpha.mp4", 1),
        _m("c10a", "/v/clip10.mp4", 2),
        _m("c2early", "/v/clip2.mp4", 3),
    ]
    assert _sort_positions(util_js, moments) == ["a", "c2early", "c2late", "c10a", "c10b"]


@pytest.mark.slow
def test_position_order_is_stable_and_keeps_same_named_files_apart(util_js) -> None:
    """#469: equal starts keep score order (stable), case variants and same-named files in two
    folders sort the same whatever the input order, and each folder's file stays together."""
    ties = [_m("first", "/v/a.mp4", 4), _m("second", "/v/a.mp4", 4), _m("third", "/v/a.mp4", 4)]
    assert _sort_positions(util_js, ties) == ["first", "second", "third"]
    mixed = [
        _m("x1", "/two/clip.mp4", 1), _m("y1", "/one/clip.mp4", 50),
        _m("x2", "/two/clip.mp4", 9), _m("y0", "/one/clip.mp4", 2),
    ]  # fmt: skip
    got = _sort_positions(util_js, mixed)
    assert _sort_positions(util_js, mixed[::-1]) == got
    assert got == ["y0", "y1", "x1", "x2"]
    cased = [_m("up", "/v/Clip.mp4", 1), _m("low", "/v/clip.mp4", 1), _m("z", "/v/zed.mp4", 0)]
    got = _sort_positions(util_js, cased)
    assert _sort_positions(util_js, cased[::-1]) == got and got[-1] == "z"


def _filter_moments(util_js, moments: list[dict], q: str, f: dict) -> list[str]:
    """Run util.js's momentMatches under node; ``f`` over NO_FILTERS."""
    ids, on = util_js(
        "const [m, q, f] = args; const all = { ...U.NO_FILTERS, ...f };"
        "return [m.filter((c) => U.momentMatches(c, q, all)).map((c) => c.id), U.filtersOn(all)];",
        moments, q, f,
    )  # fmt: skip
    assert on == any(f.get(k) not in (None, "all", False, "") for k in ("seq", "ready", "source"))
    return ids


@pytest.mark.slow
def test_moment_filters_combine_with_the_search_text(util_js) -> None:
    """Sort and filter: every quick filter and the search text must all hold; the search matches
    label, transcript or file name, case ignored; no filter shows every moment."""
    ms = [
        {"id": "a", "cli_id": "h001", "label": "Jump", "transcript": "nice one", "source": "/v/a.mp4", "picked": True, "ready": True, "must_include": True},
        {"id": "b", "label": "Crash", "transcript": None, "source": "/v/b.mp4", "picked": False, "ready": True},
        {"id": "c", "label": "jump two", "source": "/v/b.mp4", "picked": False, "ready": False, "must_include": True},
    ]  # fmt: skip

    def run(q: str, f: dict) -> list[str]:
        return _filter_moments(util_js, ms, q, f)

    assert run("", {}) == ["a", "b", "c"]
    assert run("  JUMP ", {}) == ["a", "c"]
    assert run("NICE", {}) == ["a"] and run("b.mp4", {}) == ["b", "c"]
    assert run("H001", {}) == ["a"]  # the CLI id on the card (#487)
    assert run("", {"seq": "in"}) == ["a"] and run("", {"seq": "out"}) == ["b", "c"]
    assert run("", {"ready": True}) == ["a", "b"]
    assert run("", {"must": True}) == ["a", "b", "c"]  # Must include was removed: not a filter
    assert run("", {"source": "/v/b.mp4"}) == ["b", "c"]
    assert run("jump", {"seq": "out"}) == ["c"]
    assert run("jump", {"ready": True, "source": "/v/b.mp4"}) == []


# --- Color (#461): the export's --color and --color-strength ---


def color_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
) -> tuple[object, list[list[str]]]:
    """Start a clips (the sequence) or edit (a preset) job with the Color fields ``body``."""
    if action == "edit":
        res, captured, _, _ = preset_job(client, monkeypatch, tmp_path, {"kind": "clips", **body})
        return res, captured
    return peaks_job(client, monkeypatch, tmp_path, {"action": action, **body})


def _color_flags(args: list[str]) -> list[str]:
    """The ``--color`` and ``--color-strength`` flags of ``args``, each with its value."""
    out: list[str] = []
    for i, a in enumerate(args):
        if a in ("--color", "--color-strength"):
            out += args[i : i + 2]
    return out


@pytest.mark.parametrize("action", ["clips", "edit"])
@pytest.mark.parametrize(
    ("body", "passed"),
    [
        ({"color": "auto"}, ["--color", "auto"]),
        ({"color": "underwater"}, ["--color", "underwater"]),
        ({"color": "off"}, ["--color", "off"]),  # explicit: config.json's color does not apply
        ({"color": "auto", "color_strength": 0.7}, ["--color", "auto", "--color-strength", "0.7"]),
        ({"color": "underwater", "color_strength": 0}, ["--color", "underwater", "--color-strength", "0"]),
        ({"color": "auto", "color_strength": 1}, ["--color", "auto", "--color-strength", "1"]),
        ({}, []),  # not sent: the CLI applies config.json's color itself (never to gaming)
    ],
)  # fmt: skip
def test_clips_and_edit_jobs_pass_color_and_its_strength(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
    passed: list[str],
) -> None:
    res, captured = color_job(client, monkeypatch, tmp_path, action, body)
    assert res.status_code == 200, res.text
    assert _color_flags(_cli_args(captured[0])) == passed  # and the CLI parses them


@pytest.mark.parametrize("action", ["clips", "edit"])
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"color": "max"}, "color must be one of off, auto, underwater"),
        ({"color": ["auto"]}, "color must be one of"),  # not a 500
        ({"color": "auto", "color_strength": 1.5}, "color_strength must be a number from 0 to 1"),
        ({"color": "auto", "color_strength": -0.1}, "color_strength must be a number from 0 to 1"),
        ({"color": "auto", "color_strength": "0.5"}, "color_strength must be a number from 0 to 1"),
        ({"color": "off", "color_strength": 0.5}, "color_strength needs color auto or underwater"),
        ({"color": "auto", "lut": "flat.cube"}, "color cannot be combined with lut"),
        ({"color": "off", "color_strength": 0.5, "lut": "flat.cube"}, "color cannot be combined with lut"),  # a strength alone
    ],
)  # fmt: skip
def test_clips_and_edit_jobs_reject_a_bad_color(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
    message: str,
) -> None:
    res, captured = color_job(client, monkeypatch, tmp_path, action, body)
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


def test_extend_sends_no_color_it_regrades_from_the_manifest(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``extend`` has no --color: it re-applies the grade each clip's manifest entry records."""
    body = {"action": "extend", "speech_aware": True, "color": "auto"}
    res, captured = peaks_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    assert _color_flags(_cli_args(captured[0])) == []


@pytest.mark.parametrize(
    ("cfg", "shown"),
    [(None, "off"), ("auto", "auto"), ("underwater", "underwater"), ("off", "off"),
     ("pink", "off"), (3, "off"), (["auto"], "off")],
)  # fmt: skip
def test_settings_show_the_color_default(
    client: TestClient, home: Path, cfg: object, shown: str
) -> None:
    """The page's Color starts on config.json's color; unset or invalid shows Off."""
    if cfg is not None:
        set_config(home, color=cfg)
    assert client.get("/api/settings").json()["color"] == shown


def test_the_project_view_says_an_auto_project_detected_as_gaming_is_gaming(
    client: TestClient, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "gaming"
    cache.save_json(p / "highlights.json", hl)
    write(p / "ui" / "meta.json", json.dumps({"style": "auto", "source": str(tmp_path / "c.mp4")}))
    assert client.get("/api/projects/cli").json()["gaming"] is True


@pytest.mark.parametrize(("style", "gaming"), [("gaming", True), ("funny", False), (None, False)])
def test_the_project_view_says_whether_the_footage_is_gaming(
    client: TestClient, tmp_path: Path, style: str | None, gaming: bool
) -> None:
    """The page's Color starts at Off for gaming footage, as config.json's color does."""
    p, _ = cli_project(tmp_path)
    if style is not None:
        hl = cache.load_json(p / "highlights.json")
        hl["style"] = style
        cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["gaming"] is gaming


def test_the_project_view_of_a_deleted_user_style_reads_its_recorded_base(
    client: TestClient, tmp_path: Path
) -> None:
    """The style is gone from the library: the page's Color start and --must-include-crash
    default come from the built-in base highlights.json records, not from nothing."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl |= {"style": "gone", "style_base": "gaming"}
    cache.save_json(p / "highlights.json", hl)
    view = client.get("/api/projects/cli")
    assert view.status_code == 200, view.text
    assert view.json()["gaming"] is True
    assert view.json()["must_include_crash_default"] == ui.get_style("gaming").must_include_crash


def test_the_project_view_of_a_deleted_user_style_without_a_base_is_not_an_error(
    client: TestClient, tmp_path: Path
) -> None:
    """An older file (no style_base): the page loads, with the style's options unknown."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    cache.save_json(p / "highlights.json", hl | {"style": "gone"})
    view = client.get("/api/projects/cli")
    assert view.status_code == 200, view.text
    assert view.json()["gaming"] is False and view.json()["must_include_crash_default"] is None


def test_color_strength_keeps_its_room_so_the_color_slider_never_moves() -> None:
    """Regression (#518): showing Strength moved the Color slider. Hidden (Color Off), the Strength
    row keeps its box with ``visibility: hidden``, beating the page's ``[hidden]`` rule. Layout
    jsdom cannot see; showing and hiding it is in tests/js/export.test.mjs."""
    css = _web("app.css")
    hidden = _css_decls(css, ".color-strength[hidden]")
    assert hidden["display"] == "grid !important" and hidden["visibility"] == "hidden"
    assert _css_decls(css, ".color-strength")["display"] == "grid"


def _color_note(util_js, clips: list[dict]) -> str:
    """Run util.js's colorNote under node on manifest ``clips``."""
    return util_js("return U.colorNote(args[0]);", clips)


def _graded(method: str, skip: str | None) -> dict:
    return {"file": "001_a.mp4", "color": {"method": method, "strength": 1.0, "skip": skip}}


@pytest.mark.slow
def test_color_note_counts_the_clips_graded_and_why_the_others_were_not(util_js) -> None:
    """A dive exported with Color on had every clip vetoed and the page said nothing (#461
    follow-up): Export now says how many clips the grade changed and why it left the others
    as is, grouping the reasons, in a sentence that starts with a space."""
    vetoed = [_graded("underwater", "veto_clipped") for _ in range(4)]
    assert _color_note(util_js, vetoed) == " Color: no clips graded (too much highlight clipping)."
    good = [_graded("auto", s) for s in (None, "good", None, None)]
    assert _color_note(util_js, good) == (
        " Color: 3 of 4 clips graded; 1 left as is (already looked good)."
    )
    skips = ("veto_m3_drop", "veto_clipped", "veto_m3_drop", None)
    assert _color_note(util_js, [_graded("underwater", s) for s in skips]) == (
        " Color: 1 of 4 clips graded; 3 left as is"
        " (would wash out the color: 2, too much highlight clipping: 1)."
    )
    other = ("360", "log", "no_frames", "veto_cast", "veto_m3", "veto_new", "not_underwater")
    assert _color_note(util_js, [_graded("auto", s) for s in (*other, "veto_faded", "good")]) == (
        " Color: no clips graded (360 footage: 1, log footage: 1, frames could not be read: 1,"
        " would add a color cast: 1, would oversaturate: 1, veto_new: 1,"
        " did not look underwater: 1, would wash out the color: 1, already looked good: 1)."
    )
    # PR #495's veto_faded reads as veto_m3_drop does, so the two group together
    faded = [_graded("auto", s) for s in ("veto_faded", "veto_m3_drop", None)]
    assert _color_note(util_js, faded) == (
        " Color: 1 of 3 clips graded; 2 left as is (would wash out the color)."
    )
    assert _color_note(util_js, [_graded("auto", None)] * 2) == " Color: all 2 clips graded."
    assert _color_note(util_js, [_graded("auto", None)]) == " Color: 1 clip graded."
    # Color Off, an export from before --color, or a LUT grade (no ``method``): no sentence
    assert _color_note(util_js, [{"file": "a.mp4", "color": None}, {"file": "b.mp4"}]) == ""
    assert _color_note(util_js, [{"file": "a.mp4", "lut": {"file": "x.cube"}}]) == ""
    assert _color_note(util_js, []) == ""


@pytest.mark.slow
def test_color_note_counts_a_clip_at_strength_0_as_not_graded(util_js) -> None:
    """Strength 0% applies no grade with no skip reason: such a clip is not counted as graded,
    and an export whose every clip had Strength 0% gets no sentence."""

    def at(strength: float, skip: str | None = None) -> dict:
        return {"file": "a.mp4", "color": {"method": "auto", "strength": strength, "skip": skip}}

    assert _color_note(util_js, [at(0.0), at(0.0)]) == ""
    assert _color_note(util_js, [at(0.5), at(0.0)]) == (
        " Color: 1 of 2 clips graded; 1 left as is (strength 0%)."
    )
    assert _color_note(util_js, [at(0.5, "good"), at(0.0), at(0.0)]) == (
        " Color: no clips graded (already looked good: 1, strength 0%: 2)."
    )
    # an entry without ``strength`` is at full strength
    assert _color_note(util_js, [{"file": "a.mp4", "color": {"method": "auto"}}]) == (
        " Color: 1 clip graded."
    )


def test_export_state_shows_the_color_note_from_the_manifest(
    client: TestClient, tmp_path: Path
) -> None:
    """Contract pin: the project view passes each clip's manifest ``color`` entry to the page,
    and every "Latest export" sentence of the Export state carries colorNote's sentence."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "clips_ui_1").mkdir()
    entry = {"method": "auto", "strength": 1.0, "skip": "veto_clipped"}
    clip = {"file": "001_a.mp4", "color": entry}
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": [clip]})
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    view = client.get(f"/api/projects/{pid}").json()
    assert view["clips"]["clips"][0]["color"] == entry
    # the page's note from it: tests/js/export.test.mjs


# --- Subscription logins (Settings > Subscriptions) ---


@pytest.fixture
def logins(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake CLIProxyAPI and login state: no network, no daemon."""
    fake = SimpleNamespace(
        stored={},
        release=threading.Event(),
        fail="",
        cleared=[],
    )

    def start_login(provider: str) -> dict:
        return {
            "url": f"https://auth.example/authorize?p={provider}",
            "state": f"state_{provider}",
        }

    def poll_login(state: str) -> dict:
        if not fake.release.is_set():
            return {"state": "pending", "error": None}
        if fail := fake.fail:
            return {"state": "error", "error": fail}
        provider = state.split("_", 1)[1] if "_" in state else "codex"
        fake.stored[provider] = {
            "provider": provider,
            "logged_in": True,
            "email": "a@b.c",
            "org": "corp",
            "expired": False,
            "expires_ms": None,
            "error": None,
        }
        return {"state": "done", "error": None}

    def status(provider: str) -> dict:
        return fake.stored.get(
            provider,
            {
                "provider": provider,
                "logged_in": False,
                "email": None,
                "org": None,
                "expired": False,
                "expires_ms": None,
                "error": None,
            },
        )

    def statuses() -> list[dict]:
        return [status(p) for p in cliproxy.PROVIDERS]

    def logout(provider: str) -> None:
        fake.stored.pop(provider, None)
        fake.cleared.append(provider)

    def cancel_login(state: str) -> None:
        fake.cancelled = [*getattr(fake, "cancelled", []), state]

    monkeypatch.setattr(cliproxy, "start_login", start_login)
    monkeypatch.setattr(cliproxy, "poll_login", poll_login)
    monkeypatch.setattr(cliproxy, "status", status)
    monkeypatch.setattr(cliproxy, "statuses", statuses)
    monkeypatch.setattr(cliproxy, "logout", logout)
    monkeypatch.setattr(cliproxy, "cancel_login", cancel_login)
    return fake


def login_state(client: TestClient, provider: str) -> dict:
    return client.get("/api/assistant/login/status", params={"provider": provider}).json()


def test_a_subscription_login_starts_polls_conflicts_and_logs_out_without_a_token(
    client: TestClient, logins: SimpleNamespace
) -> None:
    listed = client.get("/api/assistant/logins").json()["logins"]
    assert [x["provider"] for x in listed] == ["claude-code", "codex", "antigravity"]
    assert not any(x["logged_in"] for x in listed)
    assert login_state(client, "codex") == {"state": "idle", "error": ""}

    res = client.post("/api/assistant/login", json={"provider": "codex"})
    assert res.json() == {"url": "https://auth.example/authorize?p=codex"}
    assert login_state(client, "codex")["state"] == "pending"
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 409
    assert client.post("/api/assistant/login", json={"provider": "claude-code"}).status_code == 200

    logins.release.set()
    deadline = time.time() + 5
    while login_state(client, "codex")["state"] == "pending" and time.time() < deadline:
        time.sleep(0.02)
    assert login_state(client, "codex") == {"state": "done", "error": ""}
    texts = [
        client.get("/api/assistant/logins").text,
        client.get("/api/settings").text,
        client.post("/api/assistant/logout", json={"provider": "codex"}).text,
    ]
    assert logins.cleared == ["codex"]
    assert not any("SECRET" in t for t in texts)
    assert (
        next(
            x
            for x in client.get("/api/assistant/logins").json()["logins"]
            if x["provider"] == "codex"
        )["logged_in"]
        is False
    )


def test_a_failed_subscription_login_says_why_and_can_start_again(
    client: TestClient, logins: SimpleNamespace
) -> None:
    logins.fail = "the login state did not match"
    logins.release.set()
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 200
    deadline = time.time() + 5
    while login_state(client, "codex")["state"] == "pending" and time.time() < deadline:
        time.sleep(0.02)
    assert login_state(client, "codex") == {"state": "error", "error": logins.fail}
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 200
    assert client.post("/api/assistant/login", json={"provider": "other"}).status_code == 400


def test_a_login_still_starting_is_pending_and_not_polled(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A274: while start_login waits for the proxy the state has no proxy_state; polling
    ``None`` would store the proxy's "unknown state" error over the pending login."""
    polled: list[Any] = []

    def poll_login(state: Any) -> dict:
        polled.append(state)
        return {"state": "error", "error": "unknown login state"}

    monkeypatch.setattr(cliproxy, "poll_login", poll_login)
    states = client.app.state.login_states
    states["codex"] = {"state": "pending", "started_at": time.monotonic(), "error": ""}
    assert login_state(client, "codex") == {"state": "pending", "error": ""}
    assert polled == [] and states["codex"]["state"] == "pending"
    # one that never got a proxy_state still times out, with nothing to cancel or poll
    states["codex"]["started_at"] = time.monotonic() - ui.LOGIN_TIMEOUT_S - 1
    assert login_state(client, "codex") == {"state": "error", "error": "login timed out"}
    assert polled == [] and not getattr(logins, "cancelled", [])


def offline_until_started(launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch) -> None:
    """``cliproxy.start_login`` fails like a missing proxy until the launcher has started one."""
    real = cliproxy.start_login

    def start_login(provider: str) -> dict:
        if not launcher.is_running:
            raise not_running()
        return real(provider)

    monkeypatch.setattr(cliproxy, "start_login", start_login)


def test_a_subscription_login_starts_the_proxy_when_none_runs_and_tries_again(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = client.app.state.proxies = FakeLauncher()
    offline_until_started(launcher, monkeypatch)
    res = client.post("/api/assistant/login", json={"provider": "codex"})
    assert res.status_code == 200 and res.json() == {
        "url": "https://auth.example/authorize?p=codex"
    }
    assert launcher.starts == 1 and launcher.calls == [False]  # a login always starts it
    assert login_state(client, "codex")["state"] == "pending"


def test_a_subscription_login_whose_proxy_does_not_start_is_unavailable_with_the_reason(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = client.app.state.proxies = FakeLauncher()
    launcher.error = cliproxy.ProxyError("port 18317 is already in use")
    offline_until_started(launcher, monkeypatch)
    res = client.post("/api/assistant/login", json={"provider": "codex"})
    assert res.status_code == 503, res.text
    why = "the subscription proxy did not start: port 18317 is already in use"
    assert res.json()["detail"] == why
    assert login_state(client, "codex") == {"state": "error", "error": why}


class StartedByAnother(FakeLauncher):
    """``ensure`` finds the proxy another request just started: it runs, and this call did not start it."""

    def ensure(self, *, logged_in_only: bool = False) -> bool:
        self.calls.append(logged_in_only)
        self.is_running = True
        return False


def test_a_subscription_login_retries_when_another_request_just_started_the_proxy(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = client.app.state.proxies = StartedByAnother()
    offline_until_started(launcher, monkeypatch)
    res = client.post("/api/assistant/login", json={"provider": "codex"})
    assert res.status_code == 200, res.text
    assert res.json() == {"url": "https://auth.example/authorize?p=codex"}
    assert launcher.calls == [False]


def test_a_subscription_login_waits_for_a_proxy_that_is_still_starting(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = client.app.state.proxies = FakeLauncher(running=True)
    launcher.gate = threading.Event()  # another request is starting the proxy
    real = cliproxy.start_login

    def start_login(provider: str) -> dict:
        if not launcher.gate.is_set():
            raise cliproxy.ProxyConnectionError("proxy connection failed")
        return real(provider)

    monkeypatch.setattr(cliproxy, "start_login", start_login)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/assistant/login", json={"provider": "codex"})
        assert launcher.waiting.wait(5)  # the route waits before its first try
        launcher.gate.set()
        res = pending.result(10)
    assert res.status_code == 200, res.text
    assert launcher.starts == 0


def test_a_subscription_login_with_a_running_proxy_keeps_the_original_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = client.app.state.proxies = FakeLauncher(running=True)
    why = "proxy answered 401"

    def refused(_provider: str) -> dict:
        raise cliproxy.ProxyConnectionError(why)

    monkeypatch.setattr(cliproxy, "start_login", refused)
    res = client.post("/api/assistant/login", json={"provider": "codex"})
    assert res.status_code == 503 and res.json()["detail"] == why
    assert launcher.starts == 0
    assert login_state(client, "codex") == {"state": "error", "error": why}


def test_a_subscription_login_failing_for_another_reason_is_a_server_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.app.state.proxies = FakeLauncher(running=True)

    def broken(_provider: str) -> dict:
        raise cliproxy.ProxyError("failed to start login for codex: missing url")

    monkeypatch.setattr(cliproxy, "start_login", broken)
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 500


@pytest.mark.parametrize(
    ("running", "saved", "expect"),
    [(False, True, True), (False, False, False), (True, True, False), (True, False, False)],
)
def test_the_logins_say_whether_the_page_should_start_the_proxy(
    client: TestClient, logins: SimpleNamespace, running: bool, saved: bool, expect: bool
) -> None:
    client.app.state.proxies = FakeLauncher(running=running, saved=saved)
    got = client.get("/api/assistant/logins").json()
    assert got["proxy_running"] is running
    assert got["saved_login"] is expect  # only reported while no proxy runs


def test_a_stopped_proxy_is_not_an_error_on_the_login_rows(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.app.state.proxies = FakeLauncher()
    monkeypatch.setattr(
        cliproxy,
        "statuses",
        lambda: [
            {"provider": p, "logged_in": False, "error": cliproxy.NOT_RUNNING}
            for p in cliproxy.PROVIDERS
        ],
    )
    logins = client.get("/api/assistant/logins").json()["logins"]
    assert [x["error"] for x in logins] == [None, None, None]


def test_the_proxy_route_starts_it_only_with_a_saved_login(client: TestClient) -> None:
    launcher = client.app.state.proxies = FakeLauncher(saved=False)
    res = client.post("/api/assistant/proxy")
    assert res.status_code == 200 and res.json() == {"running": False}
    assert launcher.starts == 0 and launcher.calls == [True]

    launcher.saved = True
    res = client.post("/api/assistant/proxy")
    assert res.status_code == 200 and res.json() == {"running": True}
    assert launcher.starts == 1


def test_the_proxy_route_says_why_it_did_not_start(client: TestClient) -> None:
    launcher = client.app.state.proxies = FakeLauncher(saved=True)
    launcher.error = cliproxy.ProxyError("no binary for this platform")
    res = client.post("/api/assistant/proxy")
    assert res.status_code == 503
    assert (
        res.json()["detail"] == "the subscription proxy did not start: no binary for this platform"
    )


def test_the_proxy_route_needs_the_session_token(client: TestClient) -> None:
    launcher = client.app.state.proxies = FakeLauncher(saved=True)
    res = client.post("/api/assistant/proxy", headers={"X-Cutpoint-Token": "wrong"})
    assert res.status_code == 403 and launcher.starts == 0


def test_the_app_stops_the_proxy_it_started_on_shutdown() -> None:
    app = ui.create_app(PORT)
    launcher = app.state.proxies = FakeLauncher()
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}"):
        assert launcher.shutdowns == 0
    assert launcher.shutdowns == 1


def test_login_timeout_cancellation_does_not_hold_login_lock(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 200
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + ui.LOGIN_TIMEOUT_S + 1)

    entered, release = threading.Event(), threading.Event()
    logins.cancelled = []

    def slow_cancel(state: str) -> None:
        logins.cancelled.append(state)
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr(cliproxy, "cancel_login", slow_cancel)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        expired = pool.submit(login_state, client, "codex")
        try:
            assert entered.wait(2)
            # A second endpoint request can acquire the lock while daemon cancellation is blocked.
            idle = pool.submit(login_state, client, "claude-code")
            assert idle.result(timeout=1) == {"state": "idle", "error": ""}
        finally:
            release.set()
        assert expired.result(timeout=2) == {"state": "error", "error": "login timed out"}
    assert logins.cancelled == ["state_codex"]


def test_the_media_query_token_does_not_unlock_other_routes(
    client: TestClient, tmp_path: Path
) -> None:
    """``?t=`` is for <video>/<img> URLs only; every other route needs the header."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "final.mp4").write_bytes(b"final video")
    token = {"t": client.headers["X-Cutpoint-Token"]}
    no_header = {"X-Cutpoint-Token": ""}

    assert client.get(f"/api/projects/{pid}", params=token, headers=no_header).status_code == 403
    review = client.put(f"/api/projects/{pid}/review", params=token, headers=no_header, json={})
    assert review.status_code == 403
    media = client.get(f"/api/projects/{pid}/outputs/final.mp4", params=token, headers=no_header)
    assert media.status_code == 200 and media.content == b"final video"


@pytest.mark.parametrize("sep", ["%2F", "%5C"])
def test_outputs_route_refuses_path_traversal(client: TestClient, tmp_path: Path, sep: str) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "final.mp4").write_bytes(b"x")
    for folder in (p.parent, p.parent.parent):
        (folder / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    token = client.headers["X-Cutpoint-Token"]

    for name in (f"..{sep}secret.txt", f"..{sep}..{sep}secret.txt"):
        res = client.get(f"/api/projects/{pid}/outputs/{name}?t={token}")
        assert res.status_code == 404, (name, res.status_code)
        assert "TOP-SECRET" not in res.text


@pytest.mark.parametrize(
    ("err", "error"),
    [
        (cliproxy.ProxyError("no usable keyring: credentials not stored"), None),
        (OSError(), "OSError"),  # no message: the error type still says what failed
    ],
    ids=["keyring-refused", "bare-oserror"],
)
def test_a_login_whose_poll_fails_reports_the_error_and_can_start_again(
    client: TestClient, logins: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, err, error
) -> None:
    def broken(state: str) -> dict:
        raise err

    monkeypatch.setattr(cliproxy, "poll_login", broken)
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 200

    assert login_state(client, "codex") == {"state": "error", "error": error or str(err)}
    assert client.post("/api/assistant/login", json={"provider": "codex"}).status_code == 200


# --------------------------------------------------------------------------- resume (#109)
def progress_record(
    p: Path, command: str = "analyze", status: str = "interrupted", **over: object
) -> dict[str, object]:
    """Write a ``progress.json`` into ``p`` (made if missing) as :mod:`cutpoint.runstate` does;
    a run on another host, so ``runstate.read`` takes its status as written."""
    p.mkdir(parents=True, exist_ok=True)
    rec: dict[str, object] = {
        "version": 1,
        "generator": "cutpoint test",
        "command": command,
        "argv": [command, str(p.parent / "clip.mp4"), "--out", str(p)],
        "cwd": str(p.parent),
        "options_key": "k",
        "status": status,
        "pid": 1,
        "host": "another-host",
        "started": None,
        "updated": None,
        "exit_code": 130,
        "error": None,
        "sources": [],
        "steps": {"transcribe": {"state": "failed", "done": 1, "total": 3, "pct": 40}},
        **over,
    }
    cache.save_json(p / "progress.json", rec)
    return rec


@pytest.fixture
def scratch_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """APPDATA too (conftest covers CUTPOINT_HOME and LOCALAPPDATA): the CLI children of these
    tests never see the user's real folders."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


def test_a_project_with_only_progress_json_is_listed_as_interrupted_and_opens(
    client: TestClient, scratch_appdata: None
) -> None:
    p = ui.projects_dir() / "stopped"
    progress_record(p)
    (ui.projects_dir() / "empty").mkdir()  # no record, no analysis: still not a project
    listed = client.get("/api/projects").json()["projects"]
    assert listed == [{"id": "stopped", "name": "stopped", "interrupted": True}]
    res = client.get("/api/projects/stopped")
    assert res.status_code == 200, res.text
    got = res.json()
    assert got["highlights"] is None and got["sources"] == []  # progress.json grants no media
    assert got["run"] == {"command": "analyze", "status": "interrupted", "resumable": True}


@pytest.mark.parametrize(
    ("status", "analyzed", "resumable"),
    [
        ("interrupted", True, True),
        ("low_memory", True, True),
        ("failed", False, True),  # nothing but Resume can finish a first analysis
        ("failed", True, False),
        ("done", False, False),
    ],
)
def test_resume_is_offered_for_a_stopped_run(
    tmp_path: Path, status: str, analyzed: bool, resumable: bool
) -> None:
    p = tmp_path / "p"
    progress_record(p, status=status)
    assert ui.run_state(p, analyzed) == {
        "command": "analyze",
        "status": status,
        "resumable": resumable,
    }
    assert ui.run_state(tmp_path / "none", analyzed) is None


@pytest.mark.slow
def test_a_running_record_of_a_dead_process_reads_interrupted(tmp_path: Path) -> None:
    p = tmp_path / "p"
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    progress_record(p, status="running", pid=int(proc.stdout), host=socket.gethostname())
    assert ui.run_state(p, False)["status"] == "interrupted"


@pytest.mark.slow
def test_resume_preview_runs_the_dry_run_and_reports_its_steps(
    client: TestClient, scratch_appdata: None
) -> None:
    p = ui.projects_dir() / "stopped"
    progress_record(p, sources=[str(p.parent / "clip.mp4")])  # not there: the CLI warns
    res = client.get("/api/projects/stopped/resume")
    assert res.status_code == 200, res.text
    got = res.json()
    assert (got["command"], got["status"]) == ("analyze", "interrupted")
    assert got["steps"] == {"transcribe": {"state": "failed", "done": 1, "total": 3}}
    assert got["warnings"] == ["1 source(s) of the recorded run are not there now"]


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        ({"code": "folder_busy", "message": "... delete the lock"}, 409, ui.BUSY_TEXT),
        ({"code": "not_found", "message": "No progress.json"}, 404, "no run to resume"),
        ({"code": "error", "message": "names no command"}, 409, "names no command"),
    ],
)
def test_resume_preview_turns_cli_errors_into_plain_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, error: dict, status: int, detail: str
) -> None:
    progress_record(ui.projects_dir() / "stopped")
    seen: list[dict[str, object]] = []
    envelope = {"ok": False, "command": "resume", "schema": 1, "project": None}
    envelope |= {"outputs": {}, "warnings": [], "error": error}

    def run(args: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        seen.append({"args": args, **kw})
        return subprocess.CompletedProcess(args, 1, "log line\n" + json.dumps(envelope) + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    res = client.get("/api/projects/stopped/resume")
    assert res.status_code == status and detail in res.json()["detail"]
    p = ui.projects_dir() / "stopped"
    assert seen[0]["args"] == ui.cutpoint_cmd("resume", str(p), "--dry-run", "--json")
    assert seen[0]["cwd"] == p.parent  # the projects folder, not the server's own


@pytest.mark.slow
def test_a_folder_busy_job_shows_the_plain_state_without_the_delete_advice() -> None:
    message = "Another cutpoint run (process 9) is using X. If no other run is active, delete L"
    envelope = {"ok": False, "command": "analyze", "schema": 1, "project": None, "outputs": {}}
    envelope |= {"warnings": [], "error": {"code": "folder_busy", "message": message}}
    script = (
        "import sys\n"
        "print('12:00:00 INFO starting', flush=True)\n"
        f"print('12:00:01 ERROR ' + {message!r}, file=sys.stderr, flush=True)\n"
        f"print({json.dumps(envelope)!r}, flush=True)\n"
        "raise SystemExit(1)\n"
    )
    job = ui.Jobs().start("p", "analyze", [sys.executable, "-c", script])
    assert job.done.wait(30)
    view = job.view()
    assert (view["status"], view["error"], view["error_code"]) == (
        "failed",
        ui.BUSY_TEXT,
        "folder_busy",
    )
    assert not any("delete" in ln or '"schema"' in ln for ln in view["logs"]), view["logs"]
    assert view["logs"][0].endswith("INFO starting")


@pytest.mark.slow
def test_another_failure_keeps_its_error_line_and_reports_its_code() -> None:
    envelope = {"ok": False, "command": "clips", "schema": 1, "project": None, "outputs": {}}
    envelope |= {"warnings": [], "error": {"code": "not_found", "message": "boom"}}
    script = f"print('12:00:00 ERROR boom')\nprint({json.dumps(envelope)!r})\nraise SystemExit(1)"
    job = ui.Jobs().start("p", "clips", [sys.executable, "-c", script])
    assert job.done.wait(30)
    assert (job.status, job.error, job.error_code) == ("failed", "boom", "not_found")
    assert job.result is not None and job.result["error"]["code"] == "not_found"


@pytest.mark.slow
def test_a_failed_job_takes_its_error_from_the_envelope_before_any_error_line() -> None:
    """A270: a later log line that holds "ERROR " (transcript text, a path) does not replace
    the reason the command's --json envelope gives."""
    envelope = {"ok": False, "command": "clips", "schema": 1, "project": None, "outputs": {}}
    envelope |= {"warnings": [], "error": {"code": "error", "message": "the real reason"}}
    script = (
        "print('12:00:00 ERROR the real reason')\n"
        "print('12:00:01 INFO said: ERROR 404 in the transcript')\n"
        f"print({json.dumps(envelope)!r})\nraise SystemExit(1)"
    )
    job = ui.Jobs().start("p", "clips", [sys.executable, "-c", script])
    assert job.done.wait(30)
    assert (job.status, job.error) == ("failed", "the real reason")


@pytest.mark.parametrize(
    ("envelope_error", "expected"),
    [
        ({"code": "error", "message": "from the envelope"}, "from the envelope"),
        # an ffmpeg failure's message goes on with its stderr: its first line only
        (
            {"code": "error", "message": "ffmpeg failed (exit 1):\nline a\nline b"},
            "ffmpeg failed (exit 1):",
        ),
        ({"code": "error", "message": "\n  \nfrom the envelope\nmore"}, "from the envelope"),
        ({"code": "error", "message": " \n "}, "last line"),  # only blanks: the log's line
        ({"code": "error", "message": ""}, "last line"),  # nothing to say: the log's line
        ("not an object", "last line"),
        (None, "last line"),  # stopped before its envelope
    ],
)
def test_a_job_error_text_prefers_the_envelope_message(envelope_error: Any, expected: str) -> None:
    job = ui.Job("p", "clips")
    job.add_output("12:00:00 ERROR first line")
    job.add_output("12:00:01 ERROR last line")
    if envelope_error is not None:
        job.result = {"error": envelope_error}
    assert ui.Jobs._error_text(job, 0) == expected
    assert ui.Jobs._error_text(job, job.logged) == (
        None if expected == "last line" else expected
    )  # lines before the mark belong to the step before


@pytest.mark.slow
def test_jobs_run_in_the_folder_they_are_given(tmp_path: Path) -> None:
    job = ui.Jobs().start(
        "p", "analyze", [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp_path
    )
    assert job.done.wait(30) and job.status == "completed", job.error
    assert Path(job.logs[-1]).resolve() == tmp_path.resolve()


def test_every_project_job_runs_from_the_projects_folder_with_json(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    cache.save_json(
        p / "highlights.json", {**cache.load_json(p / "highlights.json"), "style": "funny"}
    )
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    seen = JobSpy(monkeypatch, check=False).calls
    for action in ("analyze", "clips"):
        assert client.post("/api/projects/cli/jobs", json={"action": action}).status_code == 200
    assert [s["cwd"] for s in seen] == [ui.projects_dir(), ui.projects_dir()]
    for s in seen:  # --json last, and the CLI takes it there
        assert s["raw"][-1] == "--json"
        cli.build_parser().parse_args(s["raw"][3:])


def resume_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pid: str
) -> tuple[object, list[dict[str, object]]]:
    """POST a resume job of ``pid`` with ``Jobs.start`` recording what it was given."""
    seen = JobSpy(monkeypatch).calls
    return client.post(f"/api/projects/{pid}/jobs", json={"action": "resume"}), seen


def test_the_resume_job_runs_cutpoint_resume_on_the_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = ui.projects_dir() / "stopped"
    progress_record(p)
    res, seen = resume_job(client, monkeypatch, "stopped")
    assert res.status_code == 200, res.text
    assert seen[0]["action"] == "resume"
    assert seen[0]["raw"] == [*ui.cutpoint_cmd("resume", str(p)), "--json"]
    assert seen[0]["cwd"] == ui.projects_dir()


@pytest.mark.parametrize("status", ["done", "running", None])
def test_the_resume_job_is_refused_without_a_stopped_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str | None
) -> None:
    p, _ = cli_project(tmp_path)
    if status:
        progress_record(p, status=status)
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 409 and seen == []


def test_a_resumed_export_becomes_the_latest_export(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    out = p / "clips_ui_20260101-000000"
    argv = ["clips", str(p), "--segments", "s.json", "--out", str(out), "--vertical", "--json"]
    progress_record(p, command="clips", argv=argv)
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    on_result = seen[0]["on_result"]
    out.mkdir()
    cache.save_json(out / "manifest.json", {"clips": []})
    # a record that had finished meanwhile resumes nothing: nothing is listed
    on_result(SimpleNamespace(result={"outputs": {"resumed": False, "clips_dir": str(out)}}))
    assert not (p / "ui" / "exports.json").exists()
    on_result(SimpleNamespace(result={"outputs": {"resumed": True, "clips_dir": str(out)}}))
    items = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["vertical"], e["final"]) for e in items] == [(out.name, True, None)]
    # a clips folder outside the project is never listed
    other = tmp_path / "elsewhere"
    other.mkdir()
    cache.save_json(other / "manifest.json", {"clips": []})
    on_result(SimpleNamespace(result={"outputs": {"resumed": True, "clips_dir": str(other)}}))
    assert len(cache.load_json(p / "ui" / "exports.json")["exports"]) == 1


def test_a_resumed_edit_lists_its_clips_and_final_and_a_resumed_analysis_keeps_its_skip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    argv = ["edit", "clip.mp4", "--kind", "montage", "--out", str(p), "--skip", "crashes"]
    progress_record(p, command="edit", argv=[*argv, "keyframes", "--json"])
    (p / "clips").mkdir()
    cache.save_json(p / "clips" / "manifest.json", {"clips": []})
    (p / "final.mp4").write_bytes(b"x")
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    outputs = {"resumed": True, "clips_dir": str(p / "clips"), "final": str(p / "final.mp4")}
    seen[0]["on_result"](SimpleNamespace(result={"outputs": outputs}))
    items = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["vertical"], e["final"]) for e in items] == [("clips", False, "final.mp4")]
    assert cache.load_json(p / ui.ANALYSIS_FILE)["skip"] == ["keyframes", "crashes"]


@pytest.mark.slow
def test_a_project_another_process_holds_reads_busy(client: TestClient, tmp_path: Path) -> None:
    p, _ = cli_project(tmp_path)
    assert client.get("/api/projects/cli").json()["busy"] is False
    lock = folders.lock_path(p / "analysis.run")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock.write_text(json.dumps({"pid": child.pid, "host": socket.gethostname()}))
        assert client.get("/api/projects/cli").json()["busy"] is True
    finally:
        child.kill()
        child.wait()
    assert client.get("/api/projects/cli").json()["busy"] is False  # its process is gone
    lock.unlink()
    progress_record(p, status="running")  # a run on another host still going
    assert client.get("/api/projects/cli").json()["busy"] is True


@pytest.mark.slow
def test_a_clips_job_reports_the_clips_it_took_over() -> None:
    def job(outputs: dict[str, object]) -> dict[str, object]:
        envelope = {"ok": True, "command": "clips", "schema": 1, "project": None}
        envelope |= {"outputs": outputs, "warnings": [], "error": None}
        j = ui.Jobs().start(
            "p", "clips", [sys.executable, "-c", f"print({json.dumps(envelope)!r})"]
        )
        assert j.done.wait(30) and j.status == "completed", j.error
        return j.view()

    assert job({"taken_over": 2})["taken_over"] == 2
    assert "taken_over" not in job({"taken_over": 0})
    assert "taken_over" not in job({})


@pytest.mark.slow
def test_jobs_never_import_a_module_from_the_projects_folder(tmp_path: Path) -> None:
    """``python -m cutpoint`` runs there: PYTHONSAFEPATH keeps a stray json.py out of reach."""
    (tmp_path / "json.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    assert ui.job_env()["PYTHONSAFEPATH"] == "1"
    job = ui.Jobs().start(
        "p", "analyze", [sys.executable, "-c", "import json; print('real json')"], cwd=tmp_path
    )
    assert job.done.wait(30) and job.status == "completed", (job.error, list(job.logs))
    assert list(job.logs) == ["real json"]


def test_a_finished_record_alone_is_no_project(client: TestClient) -> None:
    progress_record(ui.projects_dir() / "finished", status="done")
    assert client.get("/api/projects").json()["projects"] == []
    assert client.get("/api/projects/finished").status_code == 404


def test_resume_preview_returns_the_command_line_it_replays(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_record(ui.projects_dir() / "stopped")
    argv = ["clips", "p", "--force", "--out", "D:/elsewhere"]
    envelope = {"ok": True, "command": "clips", "schema": 1, "project": None, "warnings": []}
    envelope |= {"error": None, "outputs": {"resume": {"argv": argv, "status_before": "x"}}}
    out = json.dumps(envelope) + "\n"
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, out, "")
    )
    assert client.get("/api/projects/stopped/resume").json()["argv"] == argv
    envelope["outputs"]["resume"]["argv"] = ["clips", 3]  # never anything but strings
    out = json.dumps(envelope) + "\n"
    assert client.get("/api/projects/stopped/resume").json()["argv"] == []


def test_resuming_an_export_with_a_ui_file_in_the_way_is_refused_before_it_starts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    progress_record(p, command="clips")
    (p / "ui").write_text("not a folder", encoding="utf-8")
    res, _ = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 409 and "is not a folder" in res.json()["detail"]


@pytest.mark.parametrize("command", ["analyze", "edit"])
@pytest.mark.parametrize("skip", [True, False])
def test_resuming_an_analysis_with_skip_and_a_ui_file_in_the_way_is_refused_before_it_starts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str, skip: bool
) -> None:
    """resumed() saves a recorded --skip into ui/analysis.json: a ui file refuses the job first."""
    p, _ = cli_project(tmp_path)
    argv = [command, "clip.mp4", "--out", str(p), *(["--skip", "reactions"] if skip else [])]
    progress_record(p, command=command, argv=argv)
    (p / "ui").write_text("not a folder", encoding="utf-8")
    res, _ = resume_job(client, monkeypatch, "cli")
    if skip or command == "edit":  # edit registers its export in ui/exports.json too
        assert res.status_code == 409 and "is not a folder" in res.json()["detail"]
    else:  # nothing goes into ui: the analysis may resume
        assert res.status_code == 200, res.text


def test_resuming_an_analysis_with_skip_makes_the_ui_folder_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    argv = ["analyze", "clip.mp4", "--out", str(p), "--skip", "reactions"]
    progress_record(p, argv=argv)
    res, _ = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    assert (p / "ui").is_dir()


@pytest.mark.parametrize(
    ("argv", "seg", "picks"),
    [
        (["clips", "p", "--segments", "ui/s.json", "--out", "o"], "ui/s.json", {}),
        (
            ["clips", "p", "--top", "3", "--max-total", "60", "--out", "o"],
            None,
            {"top": 3, "ids": None, "max_total": 60.0},
        ),
        (["clips", "p", "--ids", "h001", "h004", "--json"], None, {"ids": ["h001", "h004"]}),
    ],
)
def test_a_resumed_360_export_stitches_what_its_command_line_exports(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    seg: str | None,
    picks: dict[str, object],
) -> None:
    p, _ = cli_project(tmp_path)
    (p / "ui").mkdir()
    cache.save_json(p / "ui" / "meta.json", {"name": "cli", "kind": "360"})
    progress_record(p, command="clips", argv=argv)
    stitched: list[tuple[object, ...]] = []
    patch_ui(monkeypatch, "stitch_first", lambda job, *a: stitched.append(a) and None)
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    before = seen[0]["before"]  # Jobs.start(..., plan, before)
    assert before(SimpleNamespace()) is None
    project, seg_file, chosen = stitched[0]
    assert project == p
    assert seg_file == (None if seg is None else p.parent / seg)
    assert chosen == ({"top": None, "ids": None, "max_total": None} | picks if seg is None else {})


def stopped_export(p: Path, name: str, *, marker: bool = True) -> Path:
    """The hidden resume folder a stopped export of the clips folder ``name`` leaves."""
    res = p / f".{name}.cutpoint-resume"
    res.mkdir(parents=True)
    (res / "001_a.mp4").write_bytes(b"x")
    if marker:
        cache.save_json(res / "resume.json", {"version": 1, "clips": {}})
    return res


def test_a_later_export_removes_the_resume_folders_that_can_never_be_resumed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "proj"
    old = stopped_export(p, "clips_ui_20260101-101010")
    older = stopped_export(p, "clips_ui_20260101-090909-2")
    current = stopped_export(p, "clips_ui_20260102-101010")
    progress_record(
        p, "clips", argv=["clips", str(p), "--out", str(p / "clips_ui_20260102-101010")]
    )
    with caplog.at_level("INFO", logger=ui.logger.name):
        ui.prune_resume_folders(p)
    assert not old.exists() and not older.exists()
    assert current.is_dir()  # Resume still continues it
    assert caplog.text.count("Removed") == 2


def test_every_resume_folder_goes_once_the_last_run_is_done(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    old = stopped_export(p, "clips_ui_20260101-101010")
    progress_record(
        p,
        "clips",
        status="done",
        argv=["clips", str(p), "--out", str(old.parent / "clips_ui_20260101-101010")],
    )
    ui.prune_resume_folders(p)
    assert not old.exists()


def test_a_resume_folder_of_a_locked_clips_folder_or_project_stays(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    busy = stopped_export(p, "clips_ui_20260101-101010")
    free = stopped_export(p, "clips_ui_20260101-111111")
    with folders.swap_lock(p / "clips_ui_20260101-101010"):
        ui.prune_resume_folders(p)
        assert busy.is_dir() and not free.exists()
        free = stopped_export(p, "clips_ui_20260101-111111")
    with folders.swap_lock(p / runstate.LOCK_NAME):
        ui.prune_resume_folders(p)
    assert free.is_dir()


def test_folders_cutpoint_did_not_write_are_never_removed(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    keep = [
        stopped_export(p, "clips_ui_20260101-101010", marker=False),  # no resume.json
        stopped_export(p, "my_clips"),  # not a UI export folder
        stopped_export(p, "clips_ui_notes"),
    ]
    (p / ".clips_ui_20260101-121212.cutpoint-resume").write_text("a file", encoding="utf-8")
    (p / "clips_ui_20260101-131313").mkdir()  # the clips folder itself
    ui.prune_resume_folders(p)
    assert all(f.is_dir() for f in keep)
    assert (p / ".clips_ui_20260101-121212.cutpoint-resume").is_file()
    assert (p / "clips_ui_20260101-131313").is_dir()


@pytest.mark.slow
def test_the_ui_prunes_when_an_export_job_ends(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    old = stopped_export(p, "clips_ui_20260101-101010")
    jobs = ui.Jobs()
    job = jobs.start(
        "proj",
        "clips",
        [sys.executable, "-c", "pass"],
        on_end=lambda: ui.prune_resume_folders(p),
    )
    assert job.done.wait(30)
    assert not old.exists()


def test_a_linked_resume_folder_is_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "proj"
    p.mkdir()
    target = stopped_export(tmp_path / "elsewhere", "target")
    link = p / ".clips_ui_20260101-101010.cutpoint-resume"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted")
    with caplog.at_level("INFO", logger=ui.logger.name):
        ui.prune_resume_folders(p)
    assert link.is_symlink() and (target / "resume.json").is_file()
    assert "WARNING" not in caplog.text


def test_an_unreadable_progress_record_keeps_every_resume_folder(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    keep = stopped_export(p, "clips_ui_20260101-101010")
    (p / "progress.json").write_text("{", encoding="utf-8")
    ui.prune_resume_folders(p)
    assert keep.is_dir()


@pytest.mark.slow
def test_a_failed_job_still_ends_with_on_end_and_a_failing_on_end_changes_nothing() -> None:
    calls: list[int] = []
    jobs = ui.Jobs()
    job = jobs.start(
        "proj",
        "clips",
        [sys.executable, "-c", "raise SystemExit(3)"],
        on_end=lambda: calls.append(1),
    )
    assert job.done.wait(30) and calls == [1]

    def boom() -> None:
        raise RuntimeError("cleanup")

    job = jobs.start("proj", "clips", [sys.executable, "-c", "raise SystemExit(3)"], on_end=boom)
    assert job.done.wait(30)
    assert job.status == "failed" and job.error == "exit code 3"


# ---- the trim rail's audio envelope and events ----


@pytest.fixture
def audio_sources(client: TestClient, tmp_path: Path) -> tuple[str, dict[str, str], Path]:
    """A project with a 3 s clip (a 440 Hz tone for 2 s, then silence) and a 1 s clip without
    audio: (project id, source id by name, the envelope cache folder)."""
    loud, mute = tmp_path / "tone.mp4", tmp_path / "mute.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=d=3:s=160x120:r=25"),
        *("-f", "lavfi", "-i", "sine=f=440:d=2", "-af", "apad=whole_dur=3", "-t", "3"),
        *("-c:v", "mpeg4", "-c:a", "aac", str(loud)),
    )  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=160x120:r=25", "-c:v", "mpeg4", str(mute))
    pid, _ = make_project(
        client, tmp_path, {"tone.mp4": loud.read_bytes(), "mute.mp4": mute.read_bytes()}
    )
    ids = {s["name"]: s["id"] for s in client.get(f"/api/projects/{pid}").json()["sources"]}
    return pid, ids, ui.project_dir(pid) / ui.UI_DIR / "envelope"


def get_envelope(client: TestClient, url: str, **kw: object):
    """GET ``url`` until the decode it queued is done (202 while it runs)."""
    deadline = time.monotonic() + 60
    while (res := client.get(url, **kw)).status_code == 202 and time.monotonic() < deadline:
        assert res.json() == {"pending": True}
        time.sleep(0.05)
    return res


@needs_ffmpeg
def test_envelope_is_the_peak_level_per_bucket_and_cached(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, ids, folder = audio_sources
    calls: list[list[str]] = []
    real_spawn = ui.spawn
    patch_ui(
        monkeypatch, "spawn", lambda args, **kw: (calls.append(args), real_spawn(args, **kw))[1]
    )
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"

    assert client.get(url).status_code == 202  # queued: the page asks again
    first = get_envelope(client, url)
    assert first.status_code == 200, first.text
    data = first.json()
    peaks = data["peaks"]
    assert data["step"] == pytest.approx(0.05)
    assert 58 <= len(peaks) <= 62  # 3 s at 0.05 s
    assert all(0.0 <= v <= 1.0 for v in peaks)
    assert min(peaks[5:35]) > 0.1  # the tone (AAC's priming makes the very first values soft)
    assert max(peaks[-8:]) < 0.02  # the silence after it
    assert [f.suffix for f in folder.iterdir()] == [".json"]  # nothing else left behind

    assert client.get(url).json() == data
    assert len(calls) == 1  # the second request came from the cache


@needs_ffmpeg
def test_envelope_of_a_source_without_audio_is_empty(
    client: TestClient, audio_sources: tuple[str, dict[str, str], Path]
) -> None:
    pid, ids, folder = audio_sources
    res = client.get(f"/api/projects/{pid}/sources/{ids['mute.mp4']}/envelope")
    assert res.status_code == 200
    assert res.json()["peaks"] == []
    assert not folder.exists()


@needs_ffmpeg
@pytest.mark.parametrize(
    ("pid", "sid", "token", "status"),
    [
        ("ok", "0" * 12, True, 404),  # not one of the project's sources
        ("ok", "../../x", True, 404),
        ("nope", "ok", True, 404),
        ("ok", "ok", False, 403),
    ],
)
def test_envelope_accepts_only_a_source_of_the_project(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    pid: str,
    sid: str,
    token: bool,
    status: int,
) -> None:
    real_pid, ids, folder = audio_sources
    pid = real_pid if pid == "ok" else pid
    sid = ids["tone.mp4"] if sid == "ok" else sid
    res = client.get(
        f"/api/projects/{pid}/sources/{sid}/envelope",
        headers=None if token else {"X-Cutpoint-Token": ""},
    )
    assert res.status_code == status, res.text
    assert not folder.exists()


@needs_ffmpeg
@pytest.mark.slow
def test_envelope_failure_is_a_500_and_leaves_no_cache_file(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, ids, folder = audio_sources
    patch_ui(
        monkeypatch,
        "spawn",
        lambda args, **kw: subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(1)"], stdout=subprocess.PIPE
        ),
    )
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    res = get_envelope(client, url)
    assert res.status_code == 500
    assert not folder.exists() or not list(folder.iterdir())
    assert get_envelope(client, url).status_code == 500  # asked again: decoded again, fails again


@needs_ffmpeg
@pytest.mark.slow
def test_an_envelope_decode_over_the_time_limit_is_killed_and_reported(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pid, ids, folder = audio_sources
    started: list[subprocess.Popen] = []
    real_spawn = ui.spawn

    def silent(args: list[str], **kw: object) -> subprocess.Popen:
        # through the real spawn: its own process group, as kill_tree expects
        proc = real_spawn([sys.executable, "-c", "import time; time.sleep(600)"], binary=True)
        started.append(proc)
        return proc

    patch_ui(monkeypatch, "ENVELOPE_TIMEOUT_S", 0.5)
    patch_ui(monkeypatch, "spawn", silent)
    media = client.app.state.media
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    try:
        res = get_envelope(client, url)  # 202 while it runs, then the stored error
        assert res.status_code == 500, res.text
        # the page gets the generic 500; the reason is in the server log
        assert any("took longer than" in r.getMessage() for r in caplog.records)
        assert len(started) == 1
        wait_until(
            lambda: not media.procs and started[0].poll() is not None,
            PROC_WAIT_S,
            "the timed-out decode's process to end",
        )  # no process left behind
        assert not folder.exists() or not list(folder.iterdir())
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


@needs_ffmpeg
@pytest.mark.slow
def test_shutdown_stops_an_envelope_decode_in_flight(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, ids, _folder = audio_sources
    started: list[subprocess.Popen] = []
    real_spawn = ui.spawn

    def silent(args: list[str], **kw: object) -> subprocess.Popen:
        # through the real spawn: its own process group, as kill_tree expects
        proc = real_spawn([sys.executable, "-c", "import time; time.sleep(600)"], binary=True)
        started.append(proc)
        return proc

    patch_ui(monkeypatch, "spawn", silent)
    media = client.app.state.media
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    try:
        assert client.get(url).status_code == 202
        wait_until(lambda: started and media.procs, PROC_WAIT_S, "the decode to start")
        media.shutdown()
        wait_until(
            lambda: not media.procs and started[0].poll() is not None,
            PROC_WAIT_S,
            "the decode's process to end with the server",
        )
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


@needs_ffmpeg
def test_an_envelope_decode_failing_in_any_way_is_stored_not_requeued(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pid, ids, _folder = audio_sources
    spawned: list[list[str]] = []

    def refuse(project: Path, name: str) -> Path:
        raise HTTPException(409, "the folder is not ours")

    patch_ui(monkeypatch, "_ui_folder", refuse)
    patch_ui(monkeypatch, "spawn", lambda args, **kw: spawned.append(args))
    media = client.app.state.media
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    res = get_envelope(client, url)  # 202 while it runs, then the stored error
    assert res.status_code == 500, res.text
    assert any("envelope decode" in r.getMessage() for r in caplog.records)
    assert not spawned  # refused before any ffmpeg
    deadline = time.monotonic() + 30
    while media.envelopes_running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not media.envelopes_running
    assert not media.envelope_errors  # reported once, not kept


@needs_ffmpeg
def test_an_envelope_nobody_asks_for_any_more_is_dropped_from_the_queue(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid, _ids, folder = audio_sources
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    shutil.copyfile(tmp_path / "tone.mp4", first)
    shutil.copyfile(tmp_path / "tone.mp4", second)
    started, release = threading.Event(), threading.Event()
    spawned: list[list[str]] = []
    real_spawn, real_folder = ui.spawn, ui._ui_folder

    def stuck(project: Path, name: str) -> Path:
        started.set()
        release.wait(30)
        return real_folder(project, name)

    class Clock:
        """The media module's ``time``: ``monotonic()`` jumps when the test says so."""

        skip = 0.0

        def monotonic(self) -> float:
            return time.monotonic() + self.skip

        def __getattr__(self, name: str) -> object:
            return getattr(time, name)

    clock = Clock()
    monkeypatch.setattr(media_module, "time", clock)
    patch_ui(monkeypatch, "_ui_folder", stuck)
    patch_ui(
        monkeypatch, "spawn", lambda args, **kw: (spawned.append(args), real_spawn(args, **kw))[1]
    )
    media = client.app.state.media
    project = ui.project_dir(pid)
    try:
        assert media.envelope(project, str(first)) == {"pending": True}
        assert started.wait(30)  # the first holds the worker
        assert media.envelope(project, str(second)) == {"pending": True}  # queued behind it
        clock.skip = ui.ENVELOPE_STALE_S + 1  # the page moved on: nobody asks for the second
    finally:
        release.set()
    wait_until(lambda: not media.envelopes_running, 60, "the envelope worker to finish")
    assert len(spawned) == 1  # only the first was decoded
    assert len(list(folder.glob("*.json"))) == 1
    assert not media.envelope_asked


@needs_ffmpeg
def test_a_cached_envelope_answers_while_another_source_decodes(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid, ids, folder = audio_sources
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    assert get_envelope(client, url).status_code == 200  # cached
    other = tmp_path / "other.mp4"
    shutil.copyfile(tmp_path / "tone.mp4", other)
    started, release = threading.Event(), threading.Event()
    real_folder = ui._ui_folder

    def stuck(project: Path, name: str) -> Path:
        # before the decode takes the lock (it starts ffmpeg under it): other calls still run
        started.set()
        release.wait(30)
        return real_folder(project, name)

    patch_ui(monkeypatch, "_ui_folder", stuck)
    media = client.app.state.media
    try:
        assert media.envelope(ui.project_dir(pid), str(other)) == {"pending": True}
        assert started.wait(30)  # the decode holds the slot
        assert media.envelope(ui.project_dir(pid), str(other)) == {"pending": True}  # not twice
        t0 = time.monotonic()
        assert client.get(url).status_code == 200  # the cached source answers at once
        assert time.monotonic() - t0 < 5
    finally:
        release.set()
    for _ in range(600):
        if len(list(folder.glob("*.json"))) == 2:
            break
        time.sleep(0.05)
    assert len(list(folder.glob("*.json"))) == 2


def test_envelope_ffmpeg_mixes_every_audio_track() -> None:
    from cutpoint import envelope as audio_envelope  # noqa: PLC0415

    one = audio_envelope.ffmpeg_args("a.mp4")
    assert "-filter_complex" not in one
    two = audio_envelope.ffmpeg_args("a.mp4", 2)
    assert two[two.index("-filter_complex") + 1].startswith("[0:a:0][0:a:1]amix=inputs=2")
    assert two[two.index("-map") + 1] == "[a]"


def test_prune_envelopes_keeps_only_the_current_version_of_current_sources(
    client: TestClient, tmp_path: Path
) -> None:
    pid, footage = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    media = ui.Media()
    key = media.key(footage / "a.mp4")
    assert key is not None
    folder = p / ui.UI_DIR / "envelope"
    folder.mkdir(parents=True)
    names = [
        f"{key}-{ui.ENVELOPE_VERSION}.json",
        f"{key}-0.json",
        f"{'f' * 16}-{ui.ENVELOPE_VERSION}.json",
    ]
    for n in names:
        (folder / n).write_text("{}", encoding="utf-8")
    media.prune_envelopes(p, [str(footage / "a.mp4")])
    assert [f.name for f in folder.iterdir()] == names[:1]


def test_prune_envelopes_drops_the_envelope_of_a_source_that_changed(
    client: TestClient, tmp_path: Path
) -> None:
    pid, footage = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    media = ui.Media()
    source = footage / "a.mp4"
    old = media.key(source)
    folder = p / ui.UI_DIR / "envelope"
    folder.mkdir(parents=True)
    stale = folder / f"{old}-{ui.ENVELOPE_VERSION}.json"
    stale.write_text("{}", encoding="utf-8")
    st = source.stat()
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # same size, new mtime
    new = media.key(source)
    assert new != old
    fresh = folder / f"{new}-{ui.ENVELOPE_VERSION}.json"
    fresh.write_text("{}", encoding="utf-8")
    media.envelopes_pruned.discard(p)
    media.prune_envelopes(p, [str(source)])
    assert not stale.exists()
    assert fresh.exists()


def test_project_events_sig_follows_events_json(client: TestClient, tmp_path: Path) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    assert client.get(f"/api/projects/{pid}").json()["events_sig"] is None
    path = ui.project_dir(pid) / "events.json"
    path.write_text('{"events": []}', encoding="utf-8")
    first = client.get(f"/api/projects/{pid}").json()["events_sig"]
    assert first
    path.write_text('{"events": [], "x": 1}', encoding="utf-8")
    assert client.get(f"/api/projects/{pid}").json()["events_sig"] != first
    second = client.get(f"/api/projects/{pid}").json()["events_sig"]
    (ui.project_dir(pid) / "crashes.json").write_text('{"crashes": []}', encoding="utf-8")
    third = client.get(f"/api/projects/{pid}").json()["events_sig"]
    assert third and third != second  # crashes.json counts too
    path.unlink()
    assert client.get(f"/api/projects/{pid}").json()["events_sig"]  # crashes.json alone


def test_events_are_the_located_events_without_model_descriptions(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    url = f"/api/projects/{pid}/events"
    assert client.get(url).json() == {"events": []}  # no events.json
    assert client.get("/api/projects/nope/events").status_code == 404

    def one(**over: object) -> dict:
        return {
            "source": "/v/a.mp4", "start": 1.0, "end": 2.0, "kind": "audio", "type": "audio",
            "label": "Laughter", "description": "x" * 500, "confidence": 0.91234, "details": {"a": 1},
        } | over  # fmt: skip

    doc = {
        "events": [
            one(start=9.0, end=9.5, kind="ocr", type="kill", label="A>B"),
            one(),
            one(
                kind="telemetry", type="crash", label="impact 9.1 g then a stop", start=5.0, end=5.0
            ),
            one(kind="telemetry", type="coverage", label="riding"),  # a span, not an event
            one(kind="vlm", type="vlm", label="a long model sentence"),
            one(start="soon"),
            one(source=3),
            "junk",
        ]
    }
    (ui.project_dir(pid) / "events.json").write_text(json.dumps(doc), encoding="utf-8")
    events = client.get(url).json()["events"]
    assert [(e["kind"], e["start"], e["label"]) for e in events] == [
        ("audio", 1.0, "Laughter"),
        ("telemetry", 5.0, "crash"),
        ("ocr", 9.0, "A>B"),
    ]
    assert events[0] == {
        "source": "/v/a.mp4", "start": 1.0, "end": 2.0, "kind": "audio", "type": "audio",
        "label": "Laughter", "confidence": 0.912,
    }  # fmt: skip

    (ui.project_dir(pid) / "events.json").write_text("{not json", encoding="utf-8")
    assert client.get(url).json() == {"events": []}


def test_events_serve_talk_stretches_with_their_title(client: TestClient, tmp_path: Path) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    talk = {
        "source": "/v/a.mp4", "start": 10.0, "end": 24.5, "kind": "text", "type": "dialogue",
        "label": "funny", "title": "Mia drops the cup", "confidence": 0.7,
        "description": "x" * 300, "details": {"a": 1},
    }  # fmt: skip
    plain = talk | {"start": 40.0, "end": 41.0, "title": ""}
    doc = {"events": [plain, talk]}
    (ui.project_dir(pid) / "events.json").write_text(json.dumps(doc), encoding="utf-8")
    events = client.get(f"/api/projects/{pid}/events").json()["events"]
    assert events == [
        {
            "source": "/v/a.mp4", "start": 10.0, "end": 24.5, "kind": "text", "type": "dialogue",
            "label": "Mia drops the cup", "confidence": 0.7,
        },
        {
            "source": "/v/a.mp4", "start": 40.0, "end": 41.0, "kind": "text", "type": "dialogue",
            "label": "funny", "confidence": 0.7,
        },
    ]  # fmt: skip


def test_events_add_crashes_json_entries_not_already_telemetry_crashes(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    url = f"/api/projects/{pid}/events"
    src = "/v/a.mp4"
    crashes = {
        "crashes": [
            {
                "id": "c001",
                "source": src,
                "t0": 5.0,
                "t1": 7.0,
                "confidence": 0.81234,
            },  # = telemetry
            {"id": "c002", "source": src, "t0": 50.0, "t1": 52.0, "confidence": 0.6},
            {"id": "c006", "source": src, "t0": 7.5, "t1": 8.0, "confidence": 0.55},  # 1.5 s on
            {"id": "c007", "source": src, "t0": 8.5, "t1": 9.0, "confidence": 0.5},  # 2.5 s on
            {"id": "c003", "source": "/v/b.mp4", "t0": 5.0, "t1": 7.0, "confidence": 0.5},
            {"id": "c004", "source": src, "t0": "soon", "t1": 1.0},
            {"id": "c005", "source": 3, "t0": 1.0, "t1": 2.0},
            "junk",
        ]
    }
    (ui.project_dir(pid) / "crashes.json").write_text(json.dumps(crashes), encoding="utf-8")
    only = client.get(url).json()["events"]  # no events.json: all the valid entries
    assert [(e["source"], e["start"], e["end"]) for e in only] == [
        (src, 5.0, 7.0), (src, 7.5, 8.0), (src, 8.5, 9.0), (src, 50.0, 52.0), ("/v/b.mp4", 5.0, 7.0)
    ]  # fmt: skip
    assert only[0] == {
        "source": src, "start": 5.0, "end": 7.0, "kind": "telemetry", "type": "crash",
        "label": "crash", "confidence": 0.812,
    }  # fmt: skip

    tel = {
        "source": src, "start": 6.0, "end": 6.0, "kind": "telemetry", "type": "crash",
        "label": "impact", "confidence": 0.9,
    }  # fmt: skip
    (ui.project_dir(pid) / "events.json").write_text(
        json.dumps({"events": [tel]}), encoding="utf-8"
    )
    events = client.get(url).json()["events"]
    assert [(e["source"], e["start"], e["confidence"]) for e in events] == [
        (src, 6.0, 0.9),
        (src, 8.5, 0.5),
        (src, 50.0, 0.6),
        ("/v/b.mp4", 5.0, 0.5),
    ]  # fmt: skip  (c001 and the 7.5 s one are within 2 s of the telemetry crash, 8.5 s is not)

    (ui.project_dir(pid) / "crashes.json").write_text("{not json", encoding="utf-8")
    assert [e["start"] for e in client.get(url).json()["events"]] == [6.0]


def test_events_serve_only_crashes_above_the_relative_confidence_floor(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    url = f"/api/projects/{pid}/events"
    path = ui.project_dir(pid) / "crashes.json"

    def served(*confs: float | None) -> list[float]:
        rows = [
            {"source": "/v/a.mp4", "t0": 10.0 * i, "t1": 10.0 * i + 1, "confidence": c}
            for i, c in enumerate(confs)
        ]
        path.write_text(json.dumps({"crashes": rows}), encoding="utf-8")
        return [e["confidence"] for e in client.get(url).json()["events"]]

    assert served(0.9, 0.55, 0.53, 0.3) == [0.9, 0.55]  # 0.6 x 0.9 = 0.54 and up
    assert served(0.5, 0.3, 0.29) == [0.5, 0.3]  # relative to the best, not a fixed number
    assert served(0.3, 0.19, 0.1) == [0.3]  # 0.6 x 0.3 = 0.18 is under the 0.2 minimum
    assert served(0.15, 0.1) == []  # even the best is under the minimum
    assert served(0.9, None) == [0.9]  # no confidence, no marker


# ---------------------------------------------------------------- per-join transitions (#768)
def test_review_joins_are_validated_saved_kept_and_cleared(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    src = str((folder / "a.mp4").resolve())
    cache.save_json(
        ui.project_dir(pid) / "highlights.json", {"sources": [{"path": src, "duration": 10.0}]}
    )
    seg = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    url = f"/api/projects/{pid}/review"
    pair = {"a": "h001", "b": "h002", "transition": "fade"}

    def saved() -> dict:
        return client.get(f"/api/projects/{pid}").json()["review"]

    assert client.put(url, json={"segments": [seg], "joins": [pair]}).status_code == 200
    assert saved()["joins"] == [pair]  # ids need not be in the sequence: the pair keeps its choice
    for broken in (
        "fade",  # not a list
        ["fade"],  # not an object
        [{**pair, "transition": "wipe"}],
        [{**pair, "transition": "default"}],  # Default removes the entry: it is never stored
        [{**pair, "a": ""}],
        [{**pair, "b": 5}],
        [{**pair, "a": "x" * 65}],
        [pair, {**pair, "transition": "cut"}],  # the same pair twice
    ):
        res = client.put(url, json={"segments": [seg], "joins": broken})
        assert res.status_code == 400, broken
        assert saved()["joins"] == [pair]  # a refused save leaves the review
    # a page that sends no joins (the AI Chat's Apply) keeps them; Save of the sequence too
    assert client.put(url, json={"segments": []}).status_code == 200
    assert saved()["joins"] == [pair]
    # a target change keeps them
    assert client.put(f"/api/projects/{pid}/target", json={"seconds": 60}).status_code == 200
    assert saved()["joins"] == [pair] and saved()["target"] == 60.0
    # [] clears them
    assert client.put(url, json={"segments": [], "joins": []}).status_code == 200
    assert "joins" not in saved()
    assert saved()["target"] == 60.0


def _clip_folder(folder: Path, names: list[str], ids: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for n, (name, cid) in enumerate(zip(names, ids, strict=True)):
        (folder / name).write_bytes(b"x" * (n + 1))
        entries.append({"file": name, "id": cid, "size": n + 1})
    cache.save_json(folder / "manifest.json", {"clips": entries})


def test_export_joins_keeps_only_adjacent_pairs_in_the_clip_order(tmp_path: Path) -> None:
    folder = tmp_path / "clips"
    _clip_folder(folder, ["001_a.mp4", "002_b.mp4", "003_c.mp4"], ["h1", "h2", "h3"])
    joins = [
        {"a": "h1", "b": "h2", "transition": "fade"},
        {"a": "h1", "b": "h3", "transition": "fade"},  # a reorder separated them
        {"a": "h3", "b": "h2", "transition": "fade"},  # the other way round
        {"a": "h2", "b": "gone", "transition": "cut"},  # a clip no longer there
    ]
    assert ui.export_joins(folder, joins) == [
        {"from": "001_a.mp4", "to": "002_b.mp4", "transition": "fade"}
    ]
    assert ui.export_joins(folder, []) == []


def test_export_joins_follows_a_clip_renamed_during_review(tmp_path: Path) -> None:
    folder = tmp_path / "clips"
    _clip_folder(folder, ["001_a.mp4", "002_b.mp4", "003_c.mp4"], ["h1", "h2", "h3"])
    (folder / "002_b.mp4").rename(folder / "001b_b.mp4")  # moved up, as the page's reorder does
    joins = [{"a": "h2", "b": "h3", "transition": "fade"}]
    assert ui.export_joins(folder, joins) == [
        {"from": "001b_b.mp4", "to": "003_c.mp4", "transition": "fade"}
    ]


def _joins_project(client: TestClient, tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """A command-line project with two exported clips, ``h1`` and ``h2``."""
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    _clip_folder(p / "clips_ui_1", ["001_a.mp4", "002_b.mp4"], ["h1", "h2"])
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    return p, seg


def _assemble_with(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, seg: dict, joins: list, body: dict
):
    """Save ``joins`` and start an assemble job; its ``same_folder`` check runs, as the jobs
    lock does before the job starts. Returns the response and the captured CLI args."""
    res = client.put("/api/projects/cli/review", json={"segments": [seg], "joins": joins})
    assert res.status_code == 200, res.text
    captured = JobSpy(monkeypatch).args
    return client.post("/api/projects/cli/jobs", json={"action": "assemble", **body}), captured


def test_assemble_job_hands_the_joins_over_in_a_file_and_only_when_there_are_any(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = _joins_project(client, tmp_path)
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    res, captured = _assemble_with(client, monkeypatch, seg, fade, {"transition": "cut"})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])  # the CLI parses --joins
    (written,) = (p / "ui").glob("joins-*.json")
    assert Path(_after(args, "--joins")) == written
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "joins": [{"from": "001_a.mp4", "to": "002_b.mp4", "transition": "fade"}]
    }
    assert cli.read_joins(written)  # the CLI's own reader takes the file


def test_assemble_job_has_no_joins_flag_without_joins_for_these_clips(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = _joins_project(client, tmp_path)
    res, captured = _assemble_with(client, monkeypatch, seg, [], {})
    assert res.status_code == 200, res.text
    assert "--joins" not in captured[0]
    # a pair that is not these clips' neighbours in this order counts as none
    elsewhere = [{"a": "h2", "b": "h1", "transition": "fade"}]
    res, captured = _assemble_with(client, monkeypatch, seg, elsewhere, {})
    assert res.status_code == 200, res.text
    assert "--joins" not in captured[0] and not list((p / "ui").glob("joins-*.json"))


def test_assemble_job_refuses_a_joins_file_that_a_link_takes_outside_the_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = _joins_project(client, tmp_path)
    outside = tmp_path / "outside.json"
    try:
        (p / "ui" / "joins-final_ui_1.json").symlink_to(outside)
    except OSError:
        pytest.skip("no symlinks here")
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    res, captured = _assemble_with(client, monkeypatch, seg, fade, {})
    assert res.status_code == 409, res.text
    assert not captured and not outside.exists()


def test_fade_length_is_accepted_under_a_global_cut_only_when_a_join_fades(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, seg = _joins_project(client, tmp_path)
    body = {"transition": "cut", "fade_dur": 0.5}
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    res, captured = _assemble_with(client, monkeypatch, seg, fade, body)
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--fade-dur") == "0.5"
    cut = [{"a": "h1", "b": "h2", "transition": "cut"}]
    res, captured = _assemble_with(client, monkeypatch, seg, cut, body)
    assert res.status_code == 400
    assert "fade_dur needs the fade video transition" in res.json()["detail"]
    assert not captured


def test_a_fade_pair_that_is_not_a_neighbour_in_the_export_still_allows_fade_length(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The page offers Fade length from the sequence's order. A Fade pair there (h2 then h1) that
    is no neighbour pair in the exported clips (h1 then h2) used to block Assemble with a 400."""
    _, seg = _joins_project(client, tmp_path)
    elsewhere = [{"a": "h2", "b": "h1", "transition": "fade"}]
    res, captured = _assemble_with(
        client, monkeypatch, seg, elsewhere, {"transition": "cut", "fade_dur": 0.5}
    )
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert "--joins" not in args
    assert _after(args, "--fade-dur") == "0.5"
    # the reverse: a Fade join that is exported, but no fade_dur sent
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    res, captured = _assemble_with(client, monkeypatch, seg, fade, {"transition": "cut"})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert "--joins" in args and "--fade-dur" not in args


def test_assembly_audio_fade_is_never_refused_for_the_joins_of_the_export(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, seg = _joins_project(client, tmp_path)
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]  # no Cut join in the export
    body = {"transition": "fade", "audio_transition": "crossfade", "audio_fade": 0.1}
    res, captured = _assemble_with(client, monkeypatch, seg, fade, body)
    assert res.status_code == 200, res.text
    assert _after(_cli_args(captured[0]), "--audio-fade") == "0.1"


def test_clip_joins_lists_every_neighbour_pair_with_the_reviews_choice_or_none(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "clips"
    _clip_folder(folder, ["001_a.mp4", "002_b.mp4", "003_c.mp4"], ["h1", "h2", "h3"])
    (folder / "004_d.mp4").write_bytes(b"x")  # added by hand: the manifest does not list it
    joins = [
        {"a": "h1", "b": "h2", "transition": "fade"},
        {"a": "h1", "b": "h3", "transition": "cut"},
    ]
    assert ui.clip_joins(folder, joins) == [
        {"from": "001_a.mp4", "to": "002_b.mp4", "a": "h1", "b": "h2", "transition": "fade"},
        {"from": "002_b.mp4", "to": "003_c.mp4", "a": "h2", "b": "h3", "transition": None},
        {"from": "003_c.mp4", "to": "004_d.mp4", "a": "h3", "b": None, "transition": None},
    ]
    assert ui.export_joins(folder, joins) == [
        {"from": "001_a.mp4", "to": "002_b.mp4", "transition": "fade"}
    ]


def _assemble_to_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    seg: dict,
    joins: list,
    body: dict | None = None,
) -> list:
    """Save ``joins``, start an assemble job and return the ``on_success`` hook it registered."""
    res = client.put("/api/projects/cli/review", json={"segments": [seg], "joins": joins})
    assert res.status_code == 200, res.text
    hooks = JobSpy(monkeypatch, check=False).on_success
    res = client.post("/api/projects/cli/jobs", json={"action": "assemble", **(body or {})})
    assert res.status_code == 200, res.text
    return hooks


def test_a_finished_assemble_records_the_joins_it_used_for_the_page(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = _joins_project(client, tmp_path)
    write(p / "clips_ui_1" / "003_by_hand.mp4")  # no manifest id: no join to compare
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    (hook,) = _assemble_to_success(client, monkeypatch, seg, fade)
    write(p / "final_ui_1.mp4")
    hook()
    (export,) = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert export["final"] == "final_ui_1.mp4"
    assert export["final_joins"] == [{"a": "h1", "b": "h2", "transition": "fade"}]
    assert client.get("/api/projects/cli").json()["final_joins"] == export["final_joins"]
    # a Default join is recorded as the transition it came out with: the global Cut
    (p / "final_ui_1.mp4").unlink()  # so the next run takes the same name
    (hook,) = _assemble_to_success(client, monkeypatch, seg, [])
    write(p / "final_ui_1.mp4")
    hook()
    (export,) = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert export["final_joins"] == [{"a": "h1", "b": "h2", "transition": "cut"}]


def test_an_assemble_under_a_global_fade_records_fade_for_its_default_joins(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = _joins_project(client, tmp_path)
    (hook,) = _assemble_to_success(client, monkeypatch, seg, [], {"transition": "fade"})
    write(p / "final_ui_1.mp4")
    hook()
    (export,) = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert export["final_joins"] == [{"a": "h1", "b": "h2", "transition": "fade"}]
    # a join of its own wins over the global transition
    (p / "final_ui_1.mp4").unlink()
    cut = [{"a": "h1", "b": "h2", "transition": "cut"}]
    (hook,) = _assemble_to_success(client, monkeypatch, seg, cut, {"transition": "fade"})
    write(p / "final_ui_1.mp4")
    hook()
    (export,) = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert export["final_joins"] == [{"a": "h1", "b": "h2", "transition": "cut"}]


def test_an_assemble_records_a_cut_next_to_a_word_splice_whatever_was_chosen(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Clip 2 is exact (an AI Chat word splice): assemble joins it with a hard cut, so the
    joins recorded for the page say cut, not the Fade saved for the 1 -> 2 join."""
    p, seg = _joins_project(client, tmp_path)
    manifest = cache.load_json(p / "clips_ui_1" / "manifest.json")
    manifest["clips"][1]["exact"] = True
    cache.save_json(p / "clips_ui_1" / "manifest.json", manifest)
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    (hook,) = _assemble_to_success(client, monkeypatch, seg, fade, {"transition": "fade"})
    write(p / "final_ui_1.mp4")
    hook()
    (export,) = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert export["final_joins"] == [{"a": "h1", "b": "h2", "transition": "cut"}]


def test_an_assemble_that_makes_no_final_keeps_the_joins_of_the_earlier_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = _joins_project(client, tmp_path)
    before = [{"a": "h1", "b": "h2", "transition": "cut"}]
    cache.save_json(
        p / "ui" / "exports.json",
        {"exports": [{"dir": "clips_ui_1", "final": "old.mp4", "final_joins": before}]},
    )
    fade = [{"a": "h1", "b": "h2", "transition": "fade"}]
    (hook,) = _assemble_to_success(client, monkeypatch, seg, fade)
    with pytest.raises(RuntimeError, match="without a final video"):
        hook()
    (export,) = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert export["final"] == "old.mp4" and export["final_joins"] == before


def test_final_joins_are_null_for_an_export_record_without_them(
    client: TestClient, tmp_path: Path
) -> None:
    """exports.json files from before the page recorded the joins, and a final from ``edit``."""
    p, _ = _joins_project(client, tmp_path)
    assert client.get("/api/projects/cli").json()["final_joins"] is None
    write(p / "final.mp4")
    cache.save_json(
        p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1", "final": "final.mp4"}]}
    )
    view = client.get("/api/projects/cli").json()
    assert view["final_joins"] is None and view["final_current"] is True


# --- the subscription proxy for an analysis (#832) ---


def proxy_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    launcher: FakeLauncher,
    action: str = "analyze",
    style: str = "gaming",
    meta: dict | None = None,
    argv: list[str] | None = None,
    command: str = "analyze",
) -> tuple[Any, list[SimpleNamespace]]:
    """Post a job of ``action`` on a ``style`` project (``meta`` added to its ui/meta.json) with
    ``launcher`` as the proxy (a resume replays ``argv`` of a ``command`` run); the jobs
    ``Jobs.start`` was asked for (each with the notes the job's ``before`` hook added)."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "P", "source": str(src), "style": style})
    assert res.status_code == 200, res.text
    pid = res.json()["id"]
    p = ui.project_dir(pid)
    if meta:
        cache.save_json(p / "ui" / "meta.json", {**ui.read_meta(p), **meta})
    client.app.state.proxies = launcher
    if action == "resume":
        progress_record(p, command, argv=argv or [command, str(src), "--style", style])
    return post_proxy_job(client, monkeypatch, pid, action)


def post_proxy_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pid: str, action: str
) -> tuple[Any, list[SimpleNamespace]]:
    started: list[SimpleNamespace] = []

    def start(self, project_id, action, args, on_success, check, plan, before, **kw):
        check()
        job = SimpleNamespace(notes=[], view=lambda: {"status": "running"})
        job.note = lambda level, msg: job.notes.append((level, msg))
        started.append(job)
        if before is not None:
            assert before(job) is None
        return job

    monkeypatch.setattr(ui.Jobs, "start", start)
    body = {"action": action, **({"kind": "montage"} if action == "edit" else {})}
    return client.post(f"/api/projects/{pid}/jobs", json=body), started


def test_an_analysis_with_a_subscription_dialogue_model_starts_the_proxy(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, assistant={"provider": "claude-code"})
    launcher = FakeLauncher(saved=True)
    res, started = proxy_job(client, monkeypatch, tmp_path, launcher)
    assert res.status_code == 200, res.text
    assert launcher.calls == [True] and launcher.starts == 1 and started[0].notes == []


def test_an_analysis_with_a_subscription_vision_model_starts_the_proxy(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, vision=True, vision_assistant={"provider": "codex"})
    launcher = FakeLauncher(saved=True)
    res, _ = proxy_job(client, monkeypatch, tmp_path, launcher, style="funny")
    assert res.status_code == 200, res.text
    assert launcher.calls == [True] and launcher.starts == 1


def test_an_analysis_without_a_saved_login_does_not_start_the_proxy(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, assistant={"provider": "claude-code"})
    launcher = FakeLauncher(saved=False)
    res, started = proxy_job(client, monkeypatch, tmp_path, launcher)
    assert res.status_code == 200, res.text
    assert launcher.calls == [True] and launcher.starts == 0 and started[0].notes == []


@pytest.mark.parametrize(
    ("config_data", "style"),
    [
        ({}, "gaming"),  # Ollama everywhere
        ({"assistant": {"provider": "claude-code"}}, "funny"),  # no dialogue step
        ({"vision": False, "vision_assistant": {"provider": "codex"}}, "funny"),  # no rating
    ],
)
def test_an_analysis_without_a_subscription_role_leaves_the_proxy_alone(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_data: dict,
    style: str,
) -> None:
    set_config(home, **config_data)
    launcher = FakeLauncher(saved=True)
    res, _ = proxy_job(client, monkeypatch, tmp_path, launcher, style=style)
    assert res.status_code == 200, res.text
    assert launcher.calls == []


def test_an_analysis_still_runs_when_the_proxy_does_not_start_and_the_log_says_why(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, assistant={"provider": "claude-code"})
    launcher = FakeLauncher(saved=True)
    launcher.error = cliproxy.ProxyError("port 18317 is already in use")
    res, started = proxy_job(client, monkeypatch, tmp_path, launcher)
    assert res.status_code == 200, res.text
    assert started[0].notes == [
        ("WARNING", "the subscription proxy did not start: port 18317 is already in use")
    ]


def test_a_preset_edit_with_a_subscription_vision_model_starts_the_proxy(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, vision=True, vision_assistant={"provider": "codex"})
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "funny"
    cache.save_json(p / "highlights.json", hl)
    launcher = client.app.state.proxies = FakeLauncher(saved=True)
    res, started = post_proxy_job(client, monkeypatch, "cli", "edit")
    assert res.status_code == 200, res.text
    assert launcher.calls == [True] and launcher.starts == 1 and started[0].notes == []


def test_a_360_analysis_leaves_the_proxy_alone(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, vision=True, vision_assistant={"provider": "codex"})
    launcher = FakeLauncher(saved=True)
    res, _ = proxy_job(client, monkeypatch, tmp_path, launcher, meta={"kind": "360"})
    assert res.status_code == 200, res.text
    assert launcher.calls == []


def test_a_resumed_analysis_starts_the_proxy_its_recorded_command_needs(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, assistant={"provider": "claude-code"})
    launcher = FakeLauncher(saved=True)
    res, started = proxy_job(client, monkeypatch, tmp_path, launcher, action="resume")
    assert res.status_code == 200, res.text
    assert launcher.calls == [True] and launcher.starts == 1 and started[0].notes == []


@pytest.mark.parametrize(
    ("argv", "starts"),
    [
        (["analyze"], 1),  # auto: it may pick gaming, which has the dialogue step
        (["edit"], 1),
        (["edit", "--kind", "funny-moments"], 0),  # funny: no dialogue step
    ],
    ids=["analyze", "edit", "edit-funny-moments"],
)
def test_a_resumed_analysis_without_a_style_uses_the_default(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    starts: int,
) -> None:
    set_config(home, assistant={"provider": "claude-code"})
    launcher = FakeLauncher(saved=True)
    command, *rest = argv
    argv = [command, str(tmp_path / "clip.mp4"), *rest]
    res, _ = proxy_job(
        client, monkeypatch, tmp_path, launcher, action="resume", argv=argv, command=command
    )
    assert res.status_code == 200, res.text
    assert launcher.starts == starts


CHAT = {"assistant": {"provider": "claude-code"}}


@pytest.mark.parametrize(
    ("config_data", "args", "style", "want"),
    [
        ({"vision_assistant": {"provider": "codex"}}, ["--vision"], "funny", True),
        # neither --vision nor --no-vision: the vision setting of config.json
        ({"vision": True, "vision_assistant": {"provider": "codex"}}, [], "funny", True),
        # vision unset: only auto asks the vision model (the unsure-source check)
        ({"vision_assistant": {"provider": "codex"}}, [], "auto", True),
        ({"vision_assistant": {"provider": "codex"}}, [], "gaming", False),
        # an explicit Ollama option wins over a subscription Vision model and the chat model
        (
            {"vision_assistant": {"provider": "codex"}, **CHAT},
            ["--vision", "--ollama-model=m:1"],
            "gaming",
            False,
        ),
        (CHAT, ["--no-vision"], "auto", True),
        (CHAT, ["--no-vision"], "funny", False),
        (CHAT, ["--no-vision", "--skip", "reactions", "dialogue"], "gaming", False),
        (CHAT, ["--no-vision", "--skip", "reactions"], "gaming", True),
        # no --skip: analysis.skip of config.json decides
        ({**CHAT, "analysis": {"skip": ["dialogue"]}}, ["--no-vision"], "gaming", False),
    ],
)
def test_subscription_roles(
    home: Path, config_data: dict, args: list[str], style: str, want: bool
) -> None:
    set_config(home, **config_data)
    assert ui._subscription_roles(args, style) is want


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


@pytest.mark.parametrize("style", ["nope", 3, ""], ids=["unknown", "number", "empty"])
def test_an_unknown_style_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, style: object
) -> None:
    p = auto_project(tmp_path, detection(), {"style": "auto"})
    res, captured = style_job(client, monkeypatch, {"style": style})
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
