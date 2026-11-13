"""No folders of this computer in a chat turn's error text (#1017).

``errtext`` unit cases, then the turn paths that save an error as the reply in ``chat.json``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from cutpoint.assistant import errtext, lookfor, session
from cutpoint.assistant.providers import ProviderError, Reply
from test_assistant import FakeProvider, call, project, run

WIN = r"C:\Users\Travis\My Videos\highlights.json"
USER = "Travis"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (f"cannot read {WIN}", "cannot read highlights.json"),
        (f"cannot read '{WIN}'", "cannot read highlights.json"),
        (f'cannot read "{WIN}" now', "cannot read highlights.json now"),
        ("cannot read C:/Users/Travis/My Videos/a.mp4", "cannot read a.mp4"),
        ("cannot read C:/Users/Travis/a.mp4.", "cannot read a.mp4."),
        (r"failed (C:\Users\Travis\a.mp4), retry", "failed (a.mp4), retry"),
        (r"see \\server\share\x.mp4 now", "see x.mp4 now"),
        (r"PermissionError('C:\\Users\\Travis\\p\\x.json')", "PermissionError(x.json)"),
        (r"PermissionError('\\\\srv\\share\\x.json')", "PermissionError(x.json)"),
        ("open /home/u/x.json failed", "open x.json failed"),
        ("open '/home/u/my dir/x.json' failed", "open x.json failed"),
        ("open ~/Videos/x.mp4", "open x.mp4"),
        (r"open ~\Videos\x.mp4", "open x.mp4"),
        # a folder, a name without an extension, a dotfile, a bare drive or home: <folder>
        (r"in C:\Users\Travis\Videos", "in <folder>"),
        ("in /home/travis/projects/clip", "in <folder>"),
        ("in /home/u/.config", "in <folder>"),
        (r"in C:\ now", "in <folder> now"),
        ("in ~/ now", "in <folder> now"),
        (r"in C:\Users\Travis\Videos.", "in <folder>."),
    ],
)
def test_paths_become_a_bare_name_or_folder(text: str, want: str) -> None:
    got = errtext.no_paths(text)
    assert got == want
    assert USER not in got


@pytest.mark.parametrize(
    "text",
    [
        "Ollama at http://host:11434/api is down",
        "HTTPConnectionPool(host='h', port=1): Max retries, url: /api/chat (Caused by x)",
        "use and/or 1/2 of action / sport",
        "no change",
    ],
)
def test_text_that_is_not_a_path_stays(text: str) -> None:
    assert errtext.no_paths(text) == text


def test_a_file_url_keeps_no_host_folder() -> None:
    """file:/// is a path of this computer; whatever is done to it, the folders must not stay."""
    got = errtext.no_paths("cannot open file:///home/travis/clips/a.mp4 now")
    assert "travis" not in got and "clips" not in got, got


def test_error_text_is_the_kind_and_the_message_without_errno_or_paths() -> None:
    err = PermissionError(13, "Permission denied", r"C:\Users\Travis\p\highlights.json")
    assert errtext.error_text(err) == "PermissionError: Permission denied: highlights.json"
    win = OSError(2, "No such file", r"C:\Users\Travis\a.mp4")
    got = errtext.error_text(win)
    assert "[Errno" not in got and "[WinError" not in got and USER not in got
    assert errtext.error_text(RuntimeError()) == "RuntimeError"


@pytest.mark.parametrize(
    ("text", "want"),
    [
        # F1: an apostrophe in a folder name
        (r"open C:\Users\Sean O'Brien\My Videos\a.mp4 now", "open a.mp4 now"),
        (r'open "C:\Users\Sean O\'Brien\My Videos\a.mp4"', "open a.mp4"),
        (r"in C:\Users\Sean O'Brien\x", "in <folder>"),
        (r"open C:\Users\Travis's Videos\a.mp4", "open a.mp4"),
        ("open /home/o'brien/My Videos/a.mp4 now", "open a.mp4 now"),
        ('open "/home/o\'brien/My Videos/a.mp4"', "open a.mp4"),
        ("open /home/u/a.mp4's size", "open a.mp4's size"),
        # F2: other path shapes
        ("open //server/share/Travis/a.mp4 now", "open a.mp4 now"),
        (r"open \Users\Travis\a.mp4 now", "open a.mp4 now"),
        ("error:/home/travis/a.mp4", "error:a.mp4"),
    ],
)
def test_apostrophes_and_more_path_shapes(text: str, want: str) -> None:
    got = errtext.no_paths(text)
    assert got == want
    for leak in ("Brien", "brien", "Sean", "Videos", "Travis", "travis", "server"):
        assert leak not in got


@pytest.mark.parametrize(
    "text",
    [
        r"open 'C:\Users\Sean O'Brien\My Videos\a.mp4'",
        "open '/home/o'brien/My Videos/a.mp4'",
    ],
)
def test_a_single_quoted_path_with_an_apostrophe(text: str) -> None:
    got = errtext.no_paths(text)
    assert "rien" not in got and "Videos" not in got, got


@pytest.mark.parametrize(
    "text",
    [
        "see http://host/x/y and https://host:1/a/b",
        "requests: url: /api/chat failed",
        "bad host:11434/api reply",
        r"the pattern \d+ and \w did not match",
    ],
)
def test_urls_and_regex_text_stay(text: str) -> None:
    assert errtext.no_paths(text) == text


def test_many_apostrophes_do_not_make_the_cleaning_slow() -> None:
    text = "'C:\\a'b" * 20000
    start = time.perf_counter()
    errtext.no_paths(text)
    assert time.perf_counter() - start < 2


def test_a_single_quoted_path_with_very_long_parts_still_loses_its_folders() -> None:
    long = "x" * 1500
    text = f"open 'C:\\Users\\Sean {long}\\My Videos\\{long}\\a.mp4' now"
    got = errtext.no_paths(text)
    assert "Users" not in got and "Sean" not in got and "Videos" not in got, got[:80]
    assert "x" * 100 not in got and got.endswith("now")


def test_error_text_with_an_apostrophe_in_the_path() -> None:
    err = PermissionError(13, "Permission denied", r"C:\Users\Sean O'Brien\My Videos\a.mp4")
    got = errtext.error_text(err)
    assert got == "PermissionError: Permission denied: a.mp4"
    assert not any(w in got for w in ("Brien", "Videos", "Sean"))


# -- the turn ------------------------------------------------------------------------------------


def test_an_unexpected_exception_ends_the_turn_without_the_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(self: Any, args: Any) -> Any:
        raise PermissionError(13, "Permission denied", WIN)

    monkeypatch.setattr(session.Turn, "_tool_find_moments", boom)
    fake = FakeProvider([call("find_moments", query="laugh")])
    with caplog.at_level(logging.ERROR):
        turn = run(fake)
    want = "the assistant stopped: PermissionError: Permission denied: highlights.json"
    assert turn.status == "failed"
    assert turn.view()["error"] == want
    assert turn.record()["reply"] == want and turn.record()["error"] == want
    assert [e["text"] for e in turn.events if e["kind"] == "error"] == [want]
    assert USER not in str(turn.record())
    assert USER in caplog.text and "highlights.json" in caplog.text  # the log has all of it


def test_a_look_for_error_is_cleaned_for_the_model_and_the_event(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def search(self: Any, *a: Any, **k: Any) -> Any:
        raise lookfor.LookForError(r"not in lite; run C:\Users\Travis\src\cutpoint\install.ps1")

    monkeypatch.setattr(lookfor.Finder, "search", search)
    turn = session.Turn(FakeProvider([]), project(), "x")
    with caplog.at_level(logging.INFO):
        result, _ = turn._tool_look_for({"query": "sunset"})
    assert "assistant: look_for unavailable: not in lite" in caplog.text
    assert USER in caplog.text  # the log keeps the path
    ev = next(e for e in turn.events if e.get("name") == lookfor.NAME)
    assert "install.ps1" in result and USER not in result
    assert "install.ps1" in ev["error"] and USER not in str(ev)


def test_a_provider_error_with_a_path_is_cleaned_in_the_finished_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_messages: Any) -> Reply:
        raise ProviderError(f"cannot read {WIN}")

    with caplog.at_level(logging.INFO):
        turn = run(FakeProvider([fail]))
    assert f"assistant turn {turn.id} failed: cannot read {WIN}" in caplog.text  # full, in the log
    assert turn.status == "failed"
    assert turn.error == "cannot read highlights.json"
    assert turn.record()["reply"] == "cannot read highlights.json"
    assert USER not in str(turn.record())


def _saved(reply: str, error: str | None) -> dict[str, Any]:
    return {"user": "earlier", "reply": reply, "error": error, "status": "failed"}


def _sent(fake: FakeProvider) -> str:
    return str(fake.requests[0]["messages"])


def test_an_old_failed_turn_loses_its_path_when_sent_to_the_model() -> None:
    raw = f"the assistant stopped: [Errno 13] Permission denied: '{WIN}'"
    fake = FakeProvider([Reply("ok")])
    turn = session.Turn(fake, project(), "again", earlier=[_saved(raw, raw)])
    turn.start()
    assert turn.done.wait(10)
    sent = _sent(fake)
    assert "highlights.json" in sent and USER not in sent and "My Videos" not in sent


def test_a_model_answer_with_a_path_like_text_is_sent_unchanged() -> None:
    answer = "Open C:/Users/Travis/clip.mp4 or /home/u/x.json, and/or 1/2."
    fake = FakeProvider([Reply("ok")])
    turn = session.Turn(fake, project(), "again", earlier=[_saved(answer, None)])
    turn.start()
    assert turn.done.wait(10)
    assert answer in _sent(fake)
