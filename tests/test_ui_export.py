"""Export options of ``cutpoint ui``: smart edges, audio transition, clips options, joins."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import set_config, write
from cutpoint import cache, cli, ui
from ui_helpers import (
    JobSpy,
    _after,
    _cli_args,
    _css_decls,
    _Tags,
    _web,
    cli_project,
    make_project,
    peaks_job,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


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


def test_side_tabs_and_panel_headers_stay_put_while_a_panel_scrolls() -> None:
    """Regression: the tab bar and the Sequence and AI Chat headers scrolled away with a long
    panel; they are sticky (layout jsdom cannot see). revealRow's scroll padding is in
    tests/js/graphite.test.mjs."""
    css = _web("app.css")
    assert _css_decls(css, ".side-bar")["position"] == "sticky"
    heads = _css_decls(css, "#panelSequence > .col-head, #panelAssistant > .chat-head")
    assert heads["position"] == "sticky"


def test_the_column_minimums_fit_the_narrowest_three_column_window() -> None:
    """Cross-file contract (#516): prefs.js's MOM_MIN + SIDE_MIN + STAGE_MIN fit app.css's narrowest
    three-column window, and the stage track keeps STAGE_MIN. The splitters' behaviour is in
    tests/js/workspace.test.mjs."""
    js, css = _web("prefs.js"), _web("app.css")

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
