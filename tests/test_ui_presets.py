"""One-click presets, the AI Chat tab's pins and the export's color in ``cutpoint ui``."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import set_config, write
from cutpoint import cache, edit, ui
from cutpoint.assistant import session as ai_session
from ui_helpers import (
    _after,
    _cli_args,
    _css_decls,
    _Tags,
    _web,
    cli_project,
    make_project,
    peaks_job,
    preset_job,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


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
    assert {"app.js", "util.js", "api.js", "picker.js", "chat.js", "models.js", "prefs.js"} <= {
        name for name, _ in scripts
    }
    for name, js in [*scripts, *(("index.html <script>", s) for s in inline)]:
        for sink in sinks:
            assert sink not in js, (name, sink)


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
