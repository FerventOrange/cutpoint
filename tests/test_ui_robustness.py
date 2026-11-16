"""``cutpoint ui`` server robustness (#1085 L13): a broken job watcher, a broken proxy encode, an
unreadable review.json and two config.json changes at once never leave the server stuck or lose
what the user saved."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from conftest import set_config
from cutpoint import cache, config, procs, ui
from cutpoint.assistant import providers
from cutpoint.assistant import watch as ai_watch
from cutpoint.errors import FFmpegError
from test_assistant_routes import HostedFake, use
from test_assistant_routes import no_keys as no_keys  # the fixture

PORT = 8765
WAIT = 10  # seconds; only a safety net, every wait below ends on an event


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(ui, "video_encoder", lambda: ui.VideoEncoder("cpu"))
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


# -- A265: Jobs._watch -------------------------------------------------------------------------
class AliveProc:
    """A job process that is still running and says nothing."""

    def __init__(self) -> None:
        self.stdout: list[str] = []
        self.killed = False

    def poll(self) -> int | None:
        return None if not self.killed else -9

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_a_watcher_that_breaks_fails_the_job_and_the_next_job_can_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    proc = AliveProc()
    monkeypatch.setattr(ui, "spawn", lambda args, **_kw: proc)

    def kill(p: AliveProc, *_a: Any, **_kw: Any) -> None:
        p.killed = True

    monkeypatch.setattr(procs, "kill_tree", kill)

    def broken(job: ui.Job) -> int:
        raise RuntimeError("output pipe went away")

    monkeypatch.setattr(ui.Jobs, "_follow", staticmethod(broken))
    jobs = ui.Jobs()
    with caplog.at_level(logging.ERROR):
        job = jobs.start("p", "analyze", ["x"])
        assert job.done.wait(WAIT)
    assert job.status == "failed"
    assert "output pipe went away" in (job.error or "")
    assert job.view()["error"] == job.error  # the page shows it
    assert proc.killed  # left unread, it would block on its output
    assert any(r.exc_info for r in caplog.records)

    monkeypatch.setattr(ui.Jobs, "_follow", staticmethod(lambda job: 0))
    nxt = jobs.start("p", "analyze", ["x"])  # not "another job is already running"
    assert nxt.done.wait(WAIT)
    assert nxt.status == "completed"


# -- A266: proxy lanes -------------------------------------------------------------------------
@pytest.fixture
def media() -> Iterator[ui.Media]:
    m = ui.Media()
    try:
        yield m
    finally:
        m.shutdown()


def project(tmp_path: Path) -> tuple[Path, Path]:
    p, src = tmp_path / "proj", tmp_path / "a.mp4"
    p.mkdir()
    src.write_bytes(b"x")
    return p, src


def test_an_unexpected_error_in_a_window_encode_is_listed_as_failed(
    media: ui.Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, src = project(tmp_path)

    def broken(_self: ui.Media) -> None:
        raise RuntimeError("driver said no")

    monkeypatch.setattr(ui.Media, "_gpu_encoder", broken)
    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    media._build_window(win)  # raised before
    assert media.window_failures[(p, "k" * 16)] == {(0, 500): "driver said no"}
    assert not list(p.rglob("*.part.mp4"))


def test_the_window_lane_outlives_a_window_that_raises(
    media: ui.Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, src = project(tmp_path)
    built: list[int] = []
    second = threading.Event()

    def build(_self: ui.Media, win: ui.ProxyWindow) -> None:
        built.append(win.lo)
        if win.lo == 0:
            raise RuntimeError("unexpected")
        second.set()

    monkeypatch.setattr(ui.Media, "_build_window", build)
    media.window_lane = threading.Thread(target=media._window_lane, daemon=True)
    media.window_lane.start()
    with media.lock:
        media.windows_queue.append(ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, ""))
        media.windows_queue.append(ui.ProxyWindow(p, src, "k" * 16, 500, 1000, 1, ""))
        media.window_wake.notify()
    assert second.wait(WAIT)  # the lane's thread died at the first one before
    assert built == [0, 500]
    assert media.window_lane.is_alive()


def test_an_ffmpeg_error_while_picking_the_gpu_encoder_falls_back_to_libx264(
    media: ui.Media, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> ui.VideoEncoder:
        raise FFmpegError("ffmpeg -encoders failed")

    monkeypatch.setattr(ui, "video_encoder", broken)
    assert media._gpu_encoder() is None


def test_an_unexpected_error_in_a_whole_proxy_is_logged_and_recorded(
    media: ui.Media,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    p, src = project(tmp_path)

    def broken(_self: ui.Media) -> None:
        raise RuntimeError("driver said no")

    monkeypatch.setattr(ui.Media, "_gpu_encoder", broken)
    key = "k" * 16
    proxy = p / ui.UI_DIR / "proxy" / f"{key}.mp4"
    with media.lock:
        media.proxies[p, key] = "preparing"
    with caplog.at_level(logging.ERROR):
        # as prepare() does: in the pool, a raised error was stored in a Future nobody read
        media.pool.submit(media._build, p, src, proxy, key, 1).result(WAIT)
    assert media.proxies[p, key] == "failed"  # the next request asks for it again
    assert any("proxy of a.mp4 failed" in r.getMessage() and r.exc_info for r in caplog.records)
    assert not list(p.rglob("*.part.mp4"))


# -- A267: put_review --------------------------------------------------------------------------
def cli_project(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    p = ui.projects_dir() / "cli"
    p.mkdir(parents=True)
    hl = {"sources": [{"path": str(src), "duration": 10.0}], "settings": {"inputs": [str(src)]}}
    cache.save_json(p / "highlights.json", hl)
    return p, {"id": "h001", "source": str(src), "start": 1, "end": 4.5, "label": "good"}


@pytest.mark.parametrize(
    "raw", [b'{"segments": [{"id": "h00', b"\xff\xfe\x00bad"], ids=["cut-short", "not-utf8"]
)
def test_save_is_refused_over_a_review_file_that_cannot_be_read(
    client: TestClient, tmp_path: Path, raw: bytes
) -> None:
    """Its target and joins would be dropped: never written over, as by put_target."""
    p, seg = cli_project(tmp_path)
    file = p / "ui" / "review.json"
    file.parent.mkdir()
    file.write_bytes(raw)
    res = client.put("/api/projects/cli/review", json={"segments": [seg]})
    assert res.status_code == 409, res.text
    detail = res.json()["detail"]
    assert detail.startswith("ui/review.json is not valid JSON: ")
    assert detail.endswith(
        ". Fix or delete ui/review.json (it keeps the sequence's target length and joins), "
        "then save again; the sequence was not saved"
    )
    assert str(tmp_path) not in detail  # no absolute path
    assert file.read_bytes() == raw


def test_save_is_refused_while_review_json_stays_locked(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, seg = cli_project(tmp_path)
    file = p / "ui" / "review.json"
    good = '{"segments": [], "target": 120.0}'
    file.parent.mkdir()
    file.write_text(good, encoding="utf-8")
    real = Path.read_text
    tries: list[int] = []

    def locked(self: Path, *a: Any, **kw: Any) -> str:
        if self == file:
            tries.append(1)
            raise PermissionError(13, "locked")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", locked)
    monkeypatch.setattr(cache.time, "sleep", lambda _s: None)
    res = client.put("/api/projects/cli/review", json={"segments": [seg]})
    monkeypatch.undo()
    assert res.status_code == 409, res.text
    assert res.json()["detail"].startswith("ui/review.json cannot be read: locked. Fix or delete")
    assert len(tries) == cache.REPLACE_ATTEMPTS  # waited it out first
    assert file.read_text(encoding="utf-8") == good


def test_save_writes_a_missing_review_file(client: TestClient, tmp_path: Path) -> None:
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    assert cache.read_json(p / "ui" / "review.json")["segments"][0]["id"] == "h001"


# -- A268: config.json read-modify-write -------------------------------------------------------
class SpyLock:
    """``config._WRITE_LOCK`` that tells when another thread starts waiting for it."""

    def __init__(self, waiting: threading.Event) -> None:
        self.real = threading.Lock()
        self.waiting = waiting
        self.owner: int | None = None

    def __enter__(self) -> SpyLock:
        if self.owner not in (None, threading.get_ident()):
            self.waiting.set()
        self.real.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *exc: object) -> None:
        self.owner = None
        self.real.release()


def interleave(monkeypatch: pytest.MonkeyPatch, other: Any) -> list[threading.Thread]:
    """Run ``other`` in a thread at the moment the first config.json write starts, and let that
    write go on once ``other`` finished or waits for the config lock. Returns ``[its thread]``
    once it started."""
    progressed = threading.Event()
    monkeypatch.setattr(config, "_WRITE_LOCK", SpyLock(progressed), raising=False)
    real_save = cache.save_json
    fired: list[threading.Thread] = []

    def save(path: Path, *a: Any, **kw: Any) -> None:
        if path.name == "config.json" and not fired:

            def run() -> None:
                try:
                    other()
                finally:
                    progressed.set()

            t = threading.Thread(target=run, daemon=True)
            fired.append(t)
            t.start()
            assert progressed.wait(WAIT)
        real_save(path, *a, **kw)

    monkeypatch.setattr(cache, "save_json", save)
    return fired


def test_two_overlapping_section_updates_both_survive(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_config(home, assistant={"provider": "ollama"}, vision=True)
    fired = interleave(
        monkeypatch,
        lambda: config.update_section("assistant", lambda s: {**s, "model": "b-model"}),
    )
    got = config.update_section("assistant", lambda s: {**s, "consent": ["a-key"]})
    fired[0].join(WAIT)
    saved = json.loads((home / "config.json").read_text("utf-8"))
    assert saved == {
        "assistant": {"provider": "ollama", "consent": ["a-key"], "model": "b-model"},
        "vision": True,
    }
    assert got == {"provider": "ollama", "consent": ["a-key"]}  # as this change wrote it


def test_a_change_that_writes_nothing_leaves_the_file_alone(home: Path) -> None:
    set_config(home, assistant={"provider": "ollama"})
    before = (home / "config.json").read_bytes()
    assert config.update_section("assistant", lambda s: None) == {"provider": "ollama"}
    assert config.update_section("missing", lambda s: None) == {}
    assert (home / "config.json").read_bytes() == before


def test_frame_consents_of_two_projects_given_at_once_both_survive(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, home: Path, tmp_path: Path
) -> None:
    fake = use(monkeypatch, HostedFake([]))
    agreed = providers.consent_key(fake)
    frames = ai_watch.frames_key(fake)
    set_config(home, assistant={"provider": "ollama", "consent": [agreed]})
    a, _ = cli_project(tmp_path)
    b = ui.projects_dir() / "other"
    b.mkdir()
    (b / "highlights.json").write_bytes((a / "highlights.json").read_bytes())
    answers: list[int] = []

    def other() -> None:
        url = "/api/projects/other/assistant/frames-consent"
        answers.append(client.post(url, json={"key": frames}).status_code)

    fired = interleave(monkeypatch, other)
    res = client.post("/api/projects/cli/assistant/frames-consent", json={"key": frames})
    assert res.status_code == 200, res.text
    fired[0].join(WAIT)
    assert answers == [200]
    saved = json.loads((home / "config.json").read_text("utf-8"))["assistant"]
    assert saved["consent"] == [agreed]
    assert saved["frames_consent"] == {
        ai_watch.project_key(a): [frames],
        ai_watch.project_key(b): [frames],
    }


def test_an_assistant_url_is_checked_again_against_a_provider_saved_meanwhile(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """Settings checked a url against the provider it read first; another save switched to a
    subscription provider (which takes no url) before this one wrote: refused, nothing written."""
    set_config(home, assistant={"provider": "openai"})
    real = ui.check_assistant
    calls: list[str | None] = []

    def check(value: Any, section_name: str = "assistant", existing_provider: str | None = None):
        calls.append(existing_provider)
        out = real(value, section_name, existing_provider)
        if len(calls) == 1:  # the other save lands between this one's check and its write
            config.set_config_values({"assistant": {"provider": "codex"}})
        return out

    monkeypatch.setattr(ui, "check_assistant", check)
    res = client.put("/api/settings", json={"assistant": {"url": "https://llm.example.test/v1"}})
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "assistant.url cannot be customized for subscription providers"
    assert calls == ["openai", "codex"]
    assert json.loads((home / "config.json").read_text("utf-8")) == {
        "assistant": {"provider": "codex"}
    }
