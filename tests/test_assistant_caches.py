"""#1085 A758-A763: the AI Chat's per-process caches hit while their source is unchanged and
are dropped when it changes: the permission checks of the proxy's private files, the parsed
transcript, the subscription login status, chat.json and history.json, and the CLIP text
model of the visual search (dropped when idle). Synthetic data only; no request leaves the
machine."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cutpoint import cache
from cutpoint.assistant import cliproxy, context, providers, store
from cutpoint.assistant.private_files import Checked
from cutpoint.events import visual as V

# -- A758: private file checks -------------------------------------------------------------


@pytest.fixture
def checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The private file and folder checks cliproxy runs from here on (the real ones)."""
    ran: list[str] = []
    real_file, real_dir = cliproxy.private_file, cliproxy.private_directory

    def file_check(path: Path) -> os.stat_result:
        ran.append(Path(path).name)
        return real_file(path)

    def dir_check(path: Path) -> os.stat_result:
        ran.append(Path(path).name + "/")
        return real_dir(path)

    monkeypatch.setattr(cliproxy, "private_file", file_check)
    monkeypatch.setattr(cliproxy, "private_directory", dir_check)
    monkeypatch.setattr(cliproxy, "_is_daemon_owned", lambda: False)
    return ran


def test_the_keys_file_is_checked_once_until_it_is_replaced(checks: list[str]) -> None:
    cliproxy._ensure_keys(target_port=8317)
    checks.clear()  # the write checks every time
    for _ in range(3):
        assert cliproxy.port() == 8317
        assert cliproxy._read_keys_file() is not None
        cliproxy.redact("nothing secret")
    assert checks == ["cliproxy/", "keys.json"]  # one check each before the first read

    checks.clear()
    cliproxy._ensure_keys(target_port=8318)  # replaced: checked again before it is read
    checks.clear()
    assert cliproxy.port() == 8318
    assert checks[-1] == "keys.json" and set(checks) <= {"cliproxy/", "keys.json"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_a_keys_file_made_readable_by_others_is_checked_and_made_private_again(
    checks: list[str],
) -> None:
    cliproxy._ensure_keys(target_port=8317)
    cliproxy.port()
    checks.clear()
    kp = cliproxy.keys_path()
    kp.chmod(0o644)  # same file, same content: only its status change time moves
    cliproxy.port()
    assert checks == ["keys.json"]
    assert kp.stat().st_mode & 0o777 == 0o600


def test_a_failed_check_is_not_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    cliproxy._ensure_keys(target_port=8317)
    real = cliproxy.private_file

    def deny(path: Path) -> None:
        raise PermissionError("cannot protect private Windows file")

    monkeypatch.setattr(cliproxy, "private_file", deny)
    for _ in range(2):  # the denial is not cached as a pass
        with pytest.raises(PermissionError, match="cannot protect"):
            cliproxy.port()
    monkeypatch.setattr(cliproxy, "private_file", real)
    assert cliproxy.port() == 8317


def test_a_check_that_returns_no_stat_is_never_remembered(tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    ran: list[Path] = []
    seen = Checked()
    for _ in range(2):
        seen.check(f, ran.append, directory=False)
    assert ran == [f, f] and len(seen) == 0


def test_a_checked_path_is_checked_again_after_five_minutes(tmp_path: Path) -> None:
    """A change no stat field shows (a Windows DACL) is put right within CHECK_MAX_AGE_S."""
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    ran: list[Path] = []
    now = [100.0]
    seen = Checked()
    seen._clock = lambda: now[0]  # type: ignore[method-assign]

    def check(path: Path) -> os.stat_result:
        ran.append(path)
        return os.lstat(path)

    seen.check(f, check, directory=False)
    now[0] += Checked.MAX_AGE_S - 1
    seen.check(f, check, directory=False)
    assert len(ran) == 1
    now[0] += 1
    seen.check(f, check, directory=False)
    assert len(ran) == 2


# -- A760: the transcript is parsed once per text --------------------------------------------

TRANSCRIPT = "\n".join(
    [
        "# /day1/a.mp4",
        "[a.mp4 00:00:01.0-00:00:04.0] first",
        "not a transcript line",
        "[a.mp4@00:10:00 00:10:01.0-00:10:03.5] from a recovered part",
        "",
        "# /day2/a.mp4 (from 00:00:30)",
        "[a.mp4 00:00:02.0-00:00:05.0] second day",
        "[b.mp4 00:00:02.0-00:00:05.0] another file",
    ]
)


def test_a_transcript_is_parsed_once_per_text() -> None:
    context._transcript_rows.cache_clear()
    day1 = context.transcript_lines(TRANSCRIPT, "/day1/a.mp4", 0.0, 700.0, shown="a.mp4")
    day2 = context.transcript_lines(TRANSCRIPT, "/day2/a.mp4", 0.0, 9.0, shown="a.mp4 (2)")
    assert day1 == [
        "[a.mp4 00:00:01.0-00:00:04.0] first",
        "[a.mp4@00:10:00 00:10:01.0-00:10:03.5] from a recovered part",
    ]
    assert day2 == ["[a.mp4 (2) 00:00:02.0-00:00:05.0] second day"]
    info = context._transcript_rows.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    changed = TRANSCRIPT.replace("second day", "changed")
    got = context.transcript_lines(changed, "/day2/a.mp4", 0.0, 9.0, shown="a.mp4")
    assert got == ["[a.mp4 00:00:02.0-00:00:05.0] changed"]
    assert context._transcript_rows.cache_info().misses == 2


# -- A761: the subscription login status --------------------------------------------------------


class Proxy:
    """The proxy's management answers; ``asked`` counts the credential listings."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.asked = 0
        self.files: list[dict[str, Any]] = [{"name": "c.json", "provider": "codex"}]
        self.now = 1000.0
        self.during: Any = None  # called while a listing is answered
        monkeypatch.setattr(cliproxy, "_management_request", self.request)
        monkeypatch.setattr(cliproxy, "_load_keys", lambda: ("mgmt", "client"))
        monkeypatch.setattr(cliproxy, "_is_daemon_owned", lambda: True)
        monkeypatch.setattr(cliproxy, "_clock", lambda: self.now)
        monkeypatch.setattr(cliproxy, "checkpoint", lambda _d: None)
        monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", lambda: None)

    def request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        if path == "/v8/management/credentials" and method == "GET":
            self.asked += 1
            if self.during is not None:
                self.during()
            return {"files": list(self.files)}
        if path == "/v8/management/credentials" and method == "DELETE":
            self.files = [f for f in self.files if f["name"] != kw["params"]["name"]]
        if path == "/v8/management/oauth/status":
            self.files.append({"name": "a.json", "provider": "antigravity"})
            return {"status": "ok"}
        return {}


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> Proxy:
    return Proxy(monkeypatch)


def test_the_login_status_is_reused_for_a_while(proxy: Proxy) -> None:
    sub = providers.SubscriptionProvider("codex")
    assert sub.has_key and sub.has_key
    sub._need_key()
    assert proxy.asked == 1
    proxy.now += cliproxy.STATUS_TTL_S - 1
    assert cliproxy.cached_status("antigravity")["logged_in"] is False
    assert proxy.asked == 1
    proxy.now += 1  # expired: asked again
    assert sub.has_key and proxy.asked == 2


def test_a_login_or_logout_is_seen_at_once(proxy: Proxy) -> None:
    assert cliproxy.cached_status("antigravity")["logged_in"] is False
    assert cliproxy.poll_login("state")["state"] == "done"
    assert cliproxy.cached_status("antigravity")["logged_in"] is True
    cliproxy.logout("codex")
    assert cliproxy.cached_status("codex")["logged_in"] is False


def test_the_logins_list_always_asks_and_refreshes_the_status(proxy: Proxy) -> None:
    assert cliproxy.cached_status("codex")["logged_in"] is True
    proxy.files = []  # a logout in another process
    assert not any(s["logged_in"] for s in cliproxy.statuses())
    assert cliproxy.cached_status("codex")["logged_in"] is False
    assert proxy.asked == 2


def test_a_cached_logout_is_asked_again_before_a_chat_is_refused(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login in another process is not refused for up to STATUS_TTL_S: the failing path asks
    the proxy once more."""
    sub = providers.SubscriptionProvider("antigravity")
    assert not sub.has_key  # cached: logged out
    proxy.files.append({"name": "a.json", "provider": "antigravity"})  # `cutpoint login`
    assert not sub.has_key  # still the cached answer
    sub._need_key()  # no raise: asked afresh
    assert proxy.asked == 2
    proxy.files = []
    proxy.now += cliproxy.STATUS_TTL_S
    with pytest.raises(providers.ProviderError, match="log"):
        sub._need_key()


def test_a_proxy_start_or_stop_forgets_the_status(proxy: Proxy) -> None:
    cliproxy.cached_status("codex")
    proc = SimpleNamespace(poll=lambda: 0)
    cliproxy.ProxyProcess(proc, cliproxy.contextlib.ExitStack()).stop()  # type: ignore[arg-type]
    cliproxy.cached_status("codex")
    assert proxy.asked == 2


def test_an_answer_asked_for_before_a_change_is_not_kept(proxy: Proxy) -> None:
    proxy.during = cliproxy.forget_statuses  # a logout lands while the listing is answered
    cliproxy.cached_status("codex")
    proxy.during = None
    cliproxy.cached_status("codex")
    assert proxy.asked == 2


def test_no_answer_or_no_cutpoint_proxy_is_asked_each_time(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*_a: Any, **_k: Any) -> dict[str, Any]:
        proxy.asked += 1
        raise cliproxy.ProxyConnectionError("refused")

    monkeypatch.setattr(cliproxy, "_management_request", refused)
    for _ in range(2):
        assert cliproxy.cached_status("codex")["error"] == cliproxy.NOT_RUNNING
    assert proxy.asked == 2
    monkeypatch.setattr(cliproxy, "_management_request", proxy.request)
    cliproxy.cached_status("codex")
    monkeypatch.setattr(cliproxy, "_is_daemon_owned", lambda: False)  # it stopped elsewhere
    cliproxy.cached_status("codex")
    assert proxy.asked == 4


# -- A762: chat.json and history.json ----------------------------------------------------------


def test_the_chat_is_read_once_while_chat_json_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / store.FOLDER).mkdir(parents=True)
    proposal = {"kind": "proposal", "id": "p1", "suggestions": [{"id": "s1"}]}
    store.add_turn(tmp_path, {"user": "a", "events": [proposal]})
    reads: list[str] = []
    load_json = cache.load_json
    monkeypatch.setattr(store.cache, "load_json", lambda f: reads.append(f.name) or load_json(f))
    assert [t["user"] for t in store.turns(tmp_path)] == ["a"]
    assert store.find_proposal(tmp_path, "s1")[1] == {"id": "s1"}
    assert store.find_proposal(tmp_path, "nope") is None
    assert reads == ["chat.json"]

    chat = tmp_path / store.FOLDER / store.CHAT  # another process (or the CLI) writes it
    cache.save_json(chat, {"turns": [{"user": "b"}]})
    assert [t["user"] for t in store.turns(tmp_path)] == ["b"]
    assert len(reads) == 2

    store.add_turn(tmp_path, {"user": "c"})  # read from memory, written at once
    assert len(reads) == 2
    assert [t["user"] for t in json.loads(chat.read_text("utf-8"))["turns"]] == ["b", "c"]
    assert [t["user"] for t in store.turns(tmp_path)] == ["b", "c"]
    assert len(reads) == 3  # the write forgot the copy: the new file is read
    store.clear_chat(tmp_path)
    assert store.turns(tmp_path) == []


def test_the_history_is_read_once_while_history_json_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / store.FOLDER).mkdir(parents=True)
    store.add_applied(tmp_path, {"id": "h1"})
    reads: list[str] = []
    load_json = cache.load_json
    monkeypatch.setattr(store.cache, "load_json", lambda f: reads.append(f.name) or load_json(f))
    for _ in range(3):
        assert [e["id"] for e in store.history(tmp_path)] == ["h1"]
    store.add_applied(tmp_path, {"id": "h2"})
    assert [e["id"] for e in store.history(tmp_path)] == ["h1", "h2"]
    assert reads == ["history.json", "history.json"]
    (tmp_path / store.FOLDER / store.HISTORY).unlink()  # gone: nothing is served from memory
    assert store.history(tmp_path) == []


# -- A763: the CLIP text model is dropped when idle -------------------------------------------


@pytest.fixture
def text_model(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake CPU text model, a fake clock and the idle timers started (not run)."""
    torch = pytest.importorskip("torch")
    seen = SimpleNamespace(loads=0, now=0.0, timers=[])

    class FakeClip:
        visual = object()

        def eval(self) -> FakeClip:
            return self

        def encode_text(self, tokens: list[str]) -> Any:
            return torch.ones(len(tokens), 4)

    def load_clip(**_k: Any) -> FakeClip:
        seen.loads += 1
        return FakeClip()

    monkeypatch.setitem(sys.modules, "open_clip", SimpleNamespace(get_tokenizer=lambda _a: list))
    monkeypatch.setattr(V, "load_clip", load_clip)
    monkeypatch.setattr(V, "_clock", lambda: seen.now)
    monkeypatch.setattr(V, "_start_timer", lambda delay, fn: seen.timers.append(delay) or delay)
    monkeypatch.setattr(V, "_text_timer", None)
    monkeypatch.setattr(V, "_TEXT_MODEL", {})
    return seen


def test_the_text_model_is_dropped_after_it_went_idle_and_loaded_again(text_model) -> None:
    seen = text_model
    assert V.text_embeddings(("sunset",), save=False).shape == (1, 4)
    assert seen.loads == 1 and seen.timers == [V.TEXT_IDLE_S]
    seen.now = 400.0
    V.text_embeddings(("beach",), save=False)  # a use restarts its idle time
    assert seen.loads == 1 and len(seen.timers) == 1  # one pending check
    seen.now = 600.0
    V._text_idle_check()  # 200 s idle: kept, checked again when it would be 600 s
    assert V._TEXT_MODEL and seen.timers[-1] == pytest.approx(400.0)
    seen.now = 1000.0
    V._text_idle_check()
    assert not V._TEXT_MODEL and V._text_timer is None
    V.text_embeddings(("sunset",), save=False)  # the next search loads it again
    assert seen.loads == 2 and V._TEXT_MODEL


def test_the_text_model_is_never_dropped_while_it_encodes(text_model) -> None:
    V.text_embeddings(("sunset",), save=False)
    text_model.now = 10 * V.TEXT_IDLE_S
    with V._TEXT_LOCK:  # what an encode holds
        check = threading.Thread(target=V._text_idle_check)
        check.start()
        check.join(0.2)
        assert check.is_alive() and V._TEXT_MODEL
    check.join(5)
    assert not V._TEXT_MODEL
