"""The ``cutpoint ui`` server: token, page, review saves, shutdown, folder openers, logins."""

from __future__ import annotations

import concurrent.futures
import json
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
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("httpx")

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from conftest import link_dir, media_info, patch_ui, set_config
from cutpoint import cache, ui
from cutpoint.assistant import cliproxy
from proxy_helpers import FakeLauncher, not_running
from ui_helpers import (
    CHAT,
    PORT,
    PROC_WAIT_S,
    FakeDialog,
    _css_decls,
    _web,
    cli_project,
    make_project,
    progress_record,
    wait_until,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


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


def test_number_in_checks_type_and_range_in_one_place() -> None:
    """A415: the request checks of whole numbers and ranges go through :func:`number_in`."""
    from cutpoint.ui.common import number_in  # noqa: PLC0415

    for v in (math.nan, math.inf, 10**400, True, "0.5", None, -0.1, 1.5):
        assert number_in(v, 0, 1) is None
    assert number_in(1, 0, 1) == 1.0 and type(number_in(1, 0, 1)) is float
    assert number_in(0.25, 0, 1) == 0.25
    for v in (0, 2.0, True, "3", 51):
        assert number_in(v, 1, 50, integer=True) is None
    assert number_in(7, 1, integer=True) == 7 and type(number_in(7, 1, integer=True)) is int
    assert number_in(10**30, 1, integer=True) == 10**30  # no upper bound by default


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
