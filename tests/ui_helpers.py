"""Doubles, helpers and fixtures shared by the UI tests: a ``Jobs.start`` spy, projects, fake
processes and job builders. The fixtures are registered in conftest's ``pytest_plugins``;
``no_nvenc`` and ``no_keyring`` are opt-in (the UI test modules use them in ``pytestmark``)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import HTTPException

from conftest import link_dir, media_info, patch_ui, write
from cutpoint import cache, cli, ui
from cutpoint.assistant import keys

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def job_args(args: list[str]) -> list[str]:
    """The CLI args of a job as ``start_job`` built them, before the ``--json`` it always adds
    (so the envelope's error code and outputs reach the job)."""
    assert args[-1] == "--json", args
    return args[:-1]


class JobSpy:
    """Stands in for ``Jobs.start``: records each job and reports it "running", starting nothing.

    ``args``: the CLI args of each job (without ``--json``); ``on_success``: the hook each job
    registered; ``calls``: everything each was given, as a dict (``action``, ``raw`` the args with
    ``--json``, ``args``, ``on_success``, ``check``, ``plan``, ``before`` and the keywords such as
    ``cwd`` and ``on_result``). All three are filled as the jobs start.
    ``check``: run the job's check after recording it, as the jobs lock does before a job starts
    (False: a start that skips it). ``refuse``: raised instead, before anything is recorded (as a
    second job is refused while one runs).
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        check: bool = True,
        refuse: HTTPException | None = None,
    ) -> None:
        self.args: list[list[str]] = []
        self.on_success: list[Callable[[], None] | None] = []
        self.calls: list[dict[str, Any]] = []
        self.check = check
        self.refuse = refuse

        def start(
            _jobs: object,
            project_id: str,
            action: str,
            args: list[str],
            on_success: Callable[[], None] | None = None,
            check: Callable[[], None] | None = None,
            plan: object = None,
            before: object = None,
            **kw: Any,
        ) -> SimpleNamespace:
            if self.refuse is not None:
                raise self.refuse
            self.args.append(job_args(args))
            self.on_success.append(on_success)
            self.calls.append(
                {
                    "action": action,
                    "raw": args,
                    "args": self.args[-1],
                    "on_success": on_success,
                    "check": check,
                    "plan": plan,
                    "before": before,
                    **kw,
                }
            )
            if self.check and check is not None:
                check()
            return SimpleNamespace(view=lambda: {"status": "running"})

        monkeypatch.setattr(ui.Jobs, "start", start)


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


@pytest.fixture
def no_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxies encode with libx264 unless a test says otherwise: no GPU probe in tests."""
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("cpu"))


@pytest.fixture
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


X3_CLIPS = {X3_00: b"", X3_10: b"", X3_LRV: b""}


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


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


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


def wait_until(cond: Callable[[], object], timeout: float = 10, what: str = "") -> None:
    """Poll ``cond`` until it holds; fails after ``timeout`` s, saying ``what`` it waited for."""
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out after {timeout:.0f} s" + (
            f": {what}" if what else ""
        )
        time.sleep(0.01)


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


def _cli_args(args: list[str]) -> list[str]:
    """``args`` without the interpreter prefix, after checking the CLI parses them as given."""
    assert args[:3] == ui.cutpoint_cmd()
    cli.build_parser().parse_args(args[3:])  # exits (SystemExit) on an option it does not know
    return args[3:]


def _after(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1, args
    return args[args.index(flag) + 1]


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


def resume_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, pid: str
) -> tuple[object, list[dict[str, object]]]:
    """POST a resume job of ``pid`` with ``Jobs.start`` recording what it was given."""
    seen = JobSpy(monkeypatch).calls
    return client.post(f"/api/projects/{pid}/jobs", json={"action": "resume"}), seen


CHAT = {"assistant": {"provider": "claude-code"}}
