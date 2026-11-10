"""``cutpoint edit``: analyze -> clips -> assemble in one command (#2)."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import cutpoint.analyze
from conftest import envelope, set_config
from cutpoint import cli, edit, progress, styles


def test_kind_overrides_replace_the_kind_defaults():
    k = edit.resolve_kind("montage", style="gaming", top=5)
    assert (k.style, k.top, k.max_total) == ("gaming", 5, 60)  # the montage's own length cap


@pytest.fixture
def pipeline(home, tmp_path, monkeypatch):
    """A source video and the heavy steps replaced by fakes that record their calls
    (``write_srt``: whether the fake assemble writes the final's .srt)."""
    video = tmp_path / "videos" / "My Clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"source bytes")
    calls: list[tuple[str, tuple, dict]] = []
    state = SimpleNamespace(video=video, calls=calls, write_srt=True, fail_assemble=None)

    def analyze(targets, style, out, **kw):
        calls.append(("analyze", (targets, getattr(style, "name", style), out), kw))
        out.mkdir(parents=True, exist_ok=True)
        (out / "highlights.json").write_text("{}", encoding="utf-8")

    def export_clips(project, **kw):
        calls.append(("clips", (project,), kw))
        out = kw.get("out_dir") or project / ("clips_vertical" if kw["vertical"] else "clips")
        out.mkdir(exist_ok=True)
        (out / "001_a.mp4").write_bytes(b"clip")
        state.listed = sorted(p.name for p in project.iterdir())  # while the clips exist
        return out

    def assemble(clips_dir, out=None, **kw):
        calls.append(("assemble", (clips_dir, out), kw))
        if out is None:
            stem = "final" + clips_dir.name[5:]
            out = clips_dir.parent / f"{stem}.{kw['output_format']}"
        out.write_bytes(b"final video")
        if state.write_srt:
            text = "1\n00:00:00,000 --> 00:00:01,000\nhi\n\n"
            out.with_suffix(".srt").write_text(text, encoding="utf-8")
        if state.fail_assemble is not None:
            raise state.fail_assemble
        return out

    monkeypatch.setattr(cutpoint.analyze, "analyze", analyze)
    monkeypatch.setattr(edit, "export_clips", export_clips)
    monkeypatch.setattr(edit, "assemble", assemble)
    return state


def test_edit_runs_the_steps_in_order_and_names_the_final(pipeline, home):
    video, calls = pipeline.video, pipeline.calls
    before = (video.read_bytes(), video.stat().st_mtime_ns)
    assert cli.main(["edit", str(video), "--kind", "funny-moments"]) == 0
    project = home / "projects" / "My Clip"
    assert [c[0] for c in calls] == ["analyze", "clips", "assemble"]
    assert calls[0][1] == ([video], "funny", project)
    assert (calls[1][2]["top"], calls[1][2]["order"]) == (8, "chronological")  # funny-moments
    assert calls[2][1] == (project / "clips", None)
    assert calls[2][2]["captions"] is True
    named = project / "My Clip-final.mp4"
    assert named.read_bytes() == b"final video"
    assert named.with_suffix(".srt").read_text(encoding="utf-8").endswith("hi\n\n")
    # nothing next to the source, and the source itself untouched
    assert sorted(p.name for p in video.parent.iterdir()) == ["My Clip.mp4"]
    assert (video.read_bytes(), video.stat().st_mtime_ns) == before


def test_edit_refuses_a_non_empty_clips_folder_before_analyzing(pipeline, home, caplog):
    video, calls = pipeline.video, pipeline.calls
    clips_dir = home / "projects" / "My Clip" / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "001_old.mp4").write_bytes(b"old")
    assert cli.main(["edit", str(video)]) == 1
    assert calls == []
    assert f"{clips_dir} is not empty; use --force to replace its clips" in caplog.text
    assert cli.main(["edit", str(video), "--force"]) == 0
    assert calls[1][2]["force"] is True


def test_edit_refuses_the_source_folder_as_project(pipeline, caplog):
    video, calls = pipeline.video, pipeline.calls
    caplog.set_level(logging.ERROR)
    assert cli.main(["edit", str(video), "--out", str(video.parent)]) == 1
    assert calls == []
    assert sorted(p.name for p in video.parent.iterdir()) == ["My Clip.mp4"]


def test_edit_without_new_subtitles_does_not_copy_a_stale_final_srt(pipeline, home, caplog):
    video = pipeline.video
    project = home / "projects" / "My Clip"
    project.mkdir(parents=True)
    (project / "final.srt").write_text("from an earlier run", encoding="utf-8")
    pipeline.write_srt = False  # no transcript this run: assemble writes no final.srt
    caplog.set_level(logging.INFO)
    assert cli.main(["edit", str(video)]) == 0
    assert not (project / "My Clip-final.srt").exists()
    assert "none (no transcript)" in caplog.text


@pytest.mark.parametrize(
    ("config", "argv", "expect"),
    [
        ({"vision": True}, [], {"vision": True, "vision_top": None, "skip": ()}),
        (
            {},
            ["--vision", "--vision-top", "12", "--ollama-url", "http://ollama.example:11434",
             "--ollama-model", "llava:7b"],
            {"vision": True, "vision_top": 12, "ollama_url": "http://ollama.example:11434",
             "ollama_model": "llava:7b"},
        ),
        ({}, ["--skip", "crashes", "reactions"], {"skip": ("crashes", "reactions")}),
    ],
    ids=["config-vision", "vision-options", "skip"],
)  # fmt: skip
def test_edit_hands_its_analyze_options_on(pipeline, home, config, argv, expect):
    """The flag/config precedence is test_config's; edit only has to hand the result on."""
    if config:
        set_config(home, **config)
    assert cli.main(["edit", str(pipeline.video), *argv]) == 0
    got = pipeline.calls[0][2]
    assert {k: got[k] for k in expect} == expect


def test_edit_of_a_dotted_folder_names_project_and_final_after_it(pipeline, home, monkeypatch):
    folder = pipeline.video.parent.rename(pipeline.video.parent.with_name("trip.2024"))
    monkeypatch.chdir(folder)
    assert cli.main(["edit", "."]) == 0
    assert (home / "projects" / "trip.2024" / "trip.2024-final.mp4").is_file()


def test_edit_events_without_the_extra_fails_before_analyzing(pipeline, monkeypatch, caplog):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a)
    )
    assert cli.main(["edit", str(pipeline.video), "--events"]) == 1
    assert pipeline.calls == []
    assert "'events' extra" in caplog.text


def test_edit_events_passes_the_ollama_options_to_the_vlm_layer(pipeline, monkeypatch):
    from cutpoint.events import cli as events_cli  # noqa: PLC0415
    from cutpoint.events import pipeline as events_pipeline  # noqa: PLC0415

    got: list[dict] = []
    monkeypatch.setattr(events_pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(events_cli, "run_args", lambda args, **kw: got.append(kw))
    monkeypatch.setattr(cutpoint.analyze, "report_highlights", lambda *a: None)
    argv = ["--events", "--ollama-url", "http://ollama.example:11434", "--ollama-model", "x:7b"]
    assert cli.main(["edit", str(pipeline.video), *argv]) == 0
    assert got == [{"ollama_url": "http://ollama.example:11434", "ollama_model": "x:7b"}]
    got.clear()
    assert cli.main(["edit", str(pipeline.video), "--events", "--force"]) == 0
    assert got == [{"ollama_url": None, "ollama_model": None}]  # config.json and env decide


@pytest.mark.parametrize(
    ("argv", "layers", "expected"),
    [
        ([], ("audio", "ocr", "vlm"), None),  # None: no --layers, the events defaults
        (["--skip", "crashes"], ("audio", "ocr", "vlm"), None),  # drops no default layer
        (["--skip", "keyframes"], ("audio", "vlm"), ["audio", "vlm"]),  # no HUD discovery (#659)
        (["--skip", "reactions"], ("ocr", "vlm"), ["ocr", "vlm"]),
        (["--skip", "reactions", "keyframes", "crashes"], ("vlm",), ["vlm"]),
        (
            ["--style", "action-cam", "--skip", "keyframes"],
            ("audio", "voice", "crash", "vlm"),
            ["audio", "voice", "crash", "vlm"],
        ),
        (
            ["--style", "action-cam", "--skip", "crashes"],
            ("audio", "voice", "visual", "vlm"),
            ["audio", "voice", "visual", "vlm"],
        ),
    ],
)
def test_edit_events_leaves_out_the_layers_of_skipped_steps(
    pipeline, monkeypatch, argv, layers, expected
):
    from cutpoint.events import cli as events_cli  # noqa: PLC0415
    from cutpoint.events import pipeline as events_pipeline  # noqa: PLC0415

    checked: list[tuple[str, ...]] = []
    ran: list[list[str] | None] = []
    monkeypatch.setattr(events_pipeline, "preflight", lambda layers: checked.append(layers))
    monkeypatch.setattr(events_cli, "run_args", lambda args, **kw: ran.append(args.layers))
    monkeypatch.setattr(cutpoint.analyze, "report_highlights", lambda *a: None)
    assert cli.main(["edit", str(pipeline.video), "--events", *argv]) == 0
    assert checked == [layers]
    assert ran == [expected]


@pytest.mark.parametrize(
    ("kind", "telemetry", "emphasis", "layers"),
    [
        ("highlights", True, "hype", ("audio", "voice", "visual", "crash", "vlm")),
        ("funny-moments", True, "funny", ("audio", "ocr", "vlm")),  # funny: not action-cam
        ("highlights", False, "hype", ("audio", "ocr", "vlm")),
    ],
)
def test_edit_style_auto_probes_before_the_events_preflight(
    pipeline, monkeypatch, kind, telemetry, emphasis, layers
):
    from cutpoint import category  # noqa: PLC0415
    from cutpoint.events import cli as events_cli  # noqa: PLC0415
    from cutpoint.events import pipeline as events_pipeline  # noqa: PLC0415

    checked: list[tuple[str, ...]] = []
    monkeypatch.setattr(events_pipeline, "preflight", lambda layers: checked.append(layers))
    monkeypatch.setattr(events_cli, "run_args", lambda args, **kw: None)
    monkeypatch.setattr(cutpoint.analyze, "report_highlights", lambda *a: None)
    monkeypatch.setattr(
        category,
        "probe_source",
        lambda src, vision=None: category.SourceProbe(str(src), 60.0, telemetry),
    )
    argv = ["edit", str(pipeline.video), "--kind", kind, "--style", "auto", "--events"]
    assert cli.main(argv) == 0
    assert checked == [layers]
    request = pipeline.calls[0][1][1]
    assert request.emphasis == emphasis


@pytest.mark.parametrize(("flag", "shared"), [([], True), (["--no-vision"], False)])
def test_edit_style_auto_shares_one_vision_client_with_its_analysis(
    pipeline, monkeypatch, flag, shared
):
    """The events preflight and analyze detect with one client (a failed request is not asked
    twice); --no-vision hands none on and asks nothing."""
    from cutpoint import category  # noqa: PLC0415
    from cutpoint.events import cli as events_cli  # noqa: PLC0415
    from cutpoint.events import pipeline as events_pipeline  # noqa: PLC0415

    seen = []
    monkeypatch.setattr(events_pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(events_cli, "run_args", lambda args, **kw: None)
    monkeypatch.setattr(cutpoint.analyze, "report_highlights", lambda *a: None)
    monkeypatch.setattr(
        category,
        "probe_source",
        lambda src, vision=None: seen.append(vision) or category.SourceProbe(str(src), 60.0, False),
    )
    argv = ["edit", str(pipeline.video), "--style", "auto", "--events", *flag]
    assert cli.main(argv) == 0
    footage = pipeline.calls[0][2]["footage"]
    assert seen == [footage]
    assert (footage is not None) is shared


def test_analyze_hands_vision_and_ollama_options_to_the_detection(tmp_path, monkeypatch):
    from cutpoint import analyze, category  # noqa: PLC0415

    class Stop(Exception):
        pass

    got = []

    def detect(sources, *a, **kw):
        got.append((a, kw))
        raise Stop

    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(category, "detect", detect)
    with pytest.raises(Stop):
        analyze.analyze(
            video,
            styles.AutoRequest("hype"),
            tmp_path / "p",
            vision=False,
            ollama_url="http://h:1",
            ollama_model="m",
        )
    assert got == [((False, "http://h:1", "m"), {"shared": None})]


def test_edit_with_an_explicit_style_never_probes(pipeline, monkeypatch):
    from cutpoint import category  # noqa: PLC0415
    from cutpoint.events import cli as events_cli  # noqa: PLC0415
    from cutpoint.events import pipeline as events_pipeline  # noqa: PLC0415

    monkeypatch.setattr(events_pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(events_cli, "run_args", lambda args, **kw: None)
    monkeypatch.setattr(cutpoint.analyze, "report_highlights", lambda *a: None)
    monkeypatch.setattr(category, "probe_source", lambda src, vision=None: pytest.fail("probed"))
    monkeypatch.setattr(category, "text_probe", lambda *a: pytest.fail("OCR check"))
    assert cli.main(["edit", str(pipeline.video), "--style", "gaming", "--events"]) == 0


def test_edit_vertical_names_its_outputs(pipeline, home):
    assert cli.main(["edit", str(pipeline.video), "--vertical"]) == 0
    project = home / "projects" / "My Clip"
    assert pipeline.calls[2][1] == (project / "clips_vertical", None)
    assert (project / "My Clip-final_vertical.mp4").read_bytes() == b"final video"
    assert (project / "My Clip-final_vertical.srt").is_file()


@pytest.mark.parametrize(
    ("out", "error", "code"),
    [
        (None, RuntimeError("bad video"), 1),
        ("projects/new/nested", RuntimeError("bad video"), 1),
        ("projects/new/nested", KeyboardInterrupt(), 130),
    ],
)
def test_edit_failed_analyze_leaves_no_new_project_folder(
    home, tmp_path, monkeypatch, out, error, code
):
    """analyze removes the folders it created (#217): the project and, for a new nested
    --out, its new parents; on an error and on Ctrl+C."""
    video = tmp_path / "videos" / "My Clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"source bytes")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", lambda **kw: None)

    reached: list[Path] = []

    def analyze_source(src, *a, **kw):
        reached.append(src)  # inside analyze's project folder, which exists by now
        raise error

    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    argv = ["edit", str(video), "--no-vision"] + (["--out", out] if out else [])
    assert cli.main(argv) == code
    assert reached == [video]
    projects = home / "projects"
    assert not projects.exists() or list(projects.iterdir()) == []


# -- progress steps ----------------------------------------------------------------------


def _progress_lines(capsys) -> list[dict]:
    err = capsys.readouterr().err
    return [
        json.loads(x.removeprefix("PROGRESS "))
        for x in err.splitlines()
        if x.startswith("PROGRESS ")
    ]


@pytest.mark.parametrize("kind", ["highlights", "clips", "funny-moments"])
def test_edit_reports_nine_steps_and_assembles_for_every_kind(
    pipeline, home, monkeypatch, capsys, kind
):
    monkeypatch.setenv("CUTPOINT_PROGRESS", "1")
    assert cli.main(["edit", str(pipeline.video), "--kind", kind]) == 0
    got = _progress_lines(capsys)
    assert {g["total"] for g in got} == {9}
    final = {g["step"]: g["state"] for g in got}
    assert list(final) == [*progress.ANALYZE_STEPS, "clips", "assemble"]
    assert final["clips"] == final["assemble"] == "done"
    assert [c[0] for c in pipeline.calls] == ["analyze", "clips", "assemble"]  # --kind clips too
    assert (home / "projects" / "My Clip" / "My Clip-final.mp4").is_file()


def test_edit_cancelled_in_clips_fails_that_step(pipeline, home, monkeypatch, capsys):
    monkeypatch.setenv("CUTPOINT_PROGRESS", "1")

    def cancel(project, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(edit, "export_clips", cancel)
    assert cli.main(["edit", str(pipeline.video), "--kind", "clips"]) == 130
    last = _progress_lines(capsys)[-1]
    assert (last["step"], last["state"], last["detail"]) == ("clips", "failed", "cancelled")


# -- --no-keep-clips and --format (#519) ---------------------------------------------------


def test_edit_no_keep_clips_assembles_temporary_clips_and_leaves_none(pipeline, home, capsys):
    project = home / "projects" / "My Clip"
    old = project / "clips" / "001_old.mp4"  # an earlier run's clips: never touched, no --force
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    code = cli.main(["edit", str(pipeline.video), "--no-keep-clips", "--json"])
    doc = envelope(capsys.readouterr().out)
    assert code == 0
    clips_call, assemble_call = pipeline.calls[1], pipeline.calls[2]
    tmp = clips_call[2]["out_dir"]
    assert tmp.parent == project and tmp.name.startswith(".assemble-tmp-")
    assert clips_call[2]["temporary"] is True
    run_lock = f".{tmp.name}.run.cutpoint-lock"  # held while the temporary clips exist
    project_lock = ".analysis.run.cutpoint-lock"  # held by the whole edit run (#109)
    listed = [run_lock, project_lock, tmp.name, "clips", "highlights.json"]
    assert pipeline.listed == sorted(listed)
    assert assemble_call[1] == (tmp, project / "final.mp4")
    assert not tmp.exists()
    assert sorted(p.name for p in project.iterdir()) == [
        "My Clip-final.mp4", "My Clip-final.srt", "clips", "final.mp4", "final.srt",
        "highlights.json",
    ]  # fmt: skip
    assert old.read_bytes() == b"old" and list(old.parent.iterdir()) == [old]
    assert [w["code"] for w in doc["warnings"]] == ["temporary_clips"]
    assert doc["outputs"]["copy"] == str(project / "My Clip-final.mp4")


@pytest.mark.parametrize("error", [RuntimeError("join failed"), KeyboardInterrupt()])
def test_edit_no_keep_clips_removes_the_temporary_clips_on_failure(pipeline, home, error):
    pipeline.fail_assemble = error
    code = cli.main(["edit", str(pipeline.video), "--no-keep-clips"])
    assert code == (130 if isinstance(error, KeyboardInterrupt) else 1)
    project = home / "projects" / "My Clip"
    assert not list(project.glob(".assemble-tmp-*"))
    assert not list(project.glob(".*cutpoint*"))


@pytest.mark.parametrize("fmt", ["fcpxml", "otio"])
def test_edit_format_keeps_the_clips_and_copies_the_timeline(pipeline, home, fmt):
    assert cli.main(["edit", str(pipeline.video), "--format", fmt]) == 0
    project = home / "projects" / "My Clip"
    assert pipeline.calls[2][1] == (project / "clips", None)
    assert pipeline.calls[2][2]["output_format"] == fmt
    assert (project / f"My Clip-final.{fmt}").read_bytes() == b"final video"
    assert (project / "My Clip-final.srt").is_file()
    assert not (project / "My Clip-final.mp4").exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--kind", "clips"], "with --kind clips the clips are the deliverable"),
        (["--format", "otio"], "the --format otio timeline references the clips"),
        (["--force"], "with --no-keep-clips there are none"),
    ],
)
def test_edit_no_keep_clips_refuses_what_needs_kept_clips(pipeline, capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        cli.main(["edit", str(pipeline.video), "--no-keep-clips", *argv])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert pipeline.calls == []
