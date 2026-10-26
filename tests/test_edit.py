"""``cutpoint edit``: analyze -> clips -> assemble in one command (#2)."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import cutpoint.analyze
from conftest import set_config
from cutpoint import cli, edit, progress


def test_kind_overrides_replace_the_kind_defaults():
    k = edit.resolve_kind("montage", style="gaming", top=5)
    assert (k.style, k.top, k.max_total, k.order, k.transition, k.captions) == (
        "gaming", 5, 60, "score", "fade", False,
    )  # fmt: skip
    assert k.loudnorm


@pytest.fixture
def pipeline(home, tmp_path, monkeypatch):
    """A source video and the heavy steps replaced by fakes that record their calls
    (``write_srt``: whether the fake assemble writes the final's .srt)."""
    video = tmp_path / "videos" / "My Clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"source bytes")
    calls: list[tuple[str, tuple, dict]] = []
    state = SimpleNamespace(video=video, calls=calls, write_srt=True)

    def analyze(targets, style, out, **kw):
        calls.append(("analyze", (targets, style.name, out), kw))
        out.mkdir(parents=True, exist_ok=True)
        (out / "highlights.json").write_text("{}", encoding="utf-8")

    def export_clips(project, **kw):
        calls.append(("clips", (project,), kw))
        out = project / ("clips_vertical" if kw["vertical"] else "clips")
        out.mkdir(exist_ok=True)
        (out / "001_a.mp4").write_bytes(b"clip")
        return out

    def assemble(clips_dir, **kw):
        calls.append(("assemble", (clips_dir,), kw))
        final = clips_dir.parent / ("final" + clips_dir.name[5:] + ".mp4")
        final.write_bytes(b"final video")
        if state.write_srt:
            text = "1\n00:00:00,000 --> 00:00:01,000\nhi\n\n"
            final.with_suffix(".srt").write_text(text, encoding="utf-8")
        return final

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
    assert calls[1][2] == {
        "top": 8, "max_total": None, "order": "chronological", "vertical": False, "force": False,
        "compress_peaks": "off",
    }  # fmt: skip
    assert calls[2][1] == (project / "clips",)
    assert calls[2][2] == {
        "transition": "cut", "loudnorm": True, "captions": True, "compress_peaks": "off",
    }  # fmt: skip
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


def test_edit_passes_the_config_vision_default_to_analyze(pipeline, home):
    """The flag/config precedence is test_config's; edit only has to hand the result on."""
    set_config(home, vision=True)
    assert cli.main(["edit", str(pipeline.video)]) == 0
    assert pipeline.calls[0][2] == {
        "vision": True, "vision_top": None, "ollama_url": None, "ollama_model": None,
        "layers": True, "skip": (),
    }  # fmt: skip


def test_edit_passes_the_vision_options_to_analyze(pipeline):
    argv = ["--vision-top", "12", "--ollama-url", "http://ollama.example:11434"]
    argv += ["--ollama-model", "llava:7b"]
    assert cli.main(["edit", str(pipeline.video), "--vision", *argv]) == 0
    assert pipeline.calls[0][2] == {
        "vision": True,
        "vision_top": 12,
        "ollama_url": "http://ollama.example:11434",
        "ollama_model": "llava:7b",
        "layers": True,
        "skip": (),
    }


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


def test_edit_vertical_names_its_outputs(pipeline, home):
    assert cli.main(["edit", str(pipeline.video), "--vertical"]) == 0
    project = home / "projects" / "My Clip"
    assert pipeline.calls[2][1] == (project / "clips_vertical",)
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
def test_edit_reports_eight_steps_and_assembles_for_every_kind(
    pipeline, home, monkeypatch, capsys, kind
):
    monkeypatch.setenv("CUTPOINT_PROGRESS", "1")
    assert cli.main(["edit", str(pipeline.video), "--kind", kind]) == 0
    got = _progress_lines(capsys)
    assert {g["total"] for g in got} == {8}
    final = {g["step"]: g["state"] for g in got}
    assert list(final) == [*progress.ANALYZE_STEPS, "clips", "assemble"]
    assert final["clips"] == final["assemble"] == "done"
    assert [c[0] for c in pipeline.calls] == ["analyze", "clips", "assemble"]  # --kind clips too
    assert (home / "projects" / "My Clip" / "My Clip-final.mp4").is_file()


def test_edit_without_the_env_var_writes_no_progress(pipeline, capsys):
    assert cli.main(["edit", str(pipeline.video)]) == 0
    assert "PROGRESS" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "skip"), [([], ()), (["--skip", "crashes", "reactions"], ("crashes", "reactions"))]
)
def test_edit_passes_skip_to_analyze(pipeline, argv, skip):
    assert cli.main(["edit", str(pipeline.video), *argv]) == 0
    assert pipeline.calls[0][2]["skip"] == skip


def test_edit_cancelled_in_clips_fails_that_step(pipeline, home, monkeypatch, capsys):
    monkeypatch.setenv("CUTPOINT_PROGRESS", "1")

    def cancel(project, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(edit, "export_clips", cancel)
    assert cli.main(["edit", str(pipeline.video), "--kind", "clips"]) == 130
    last = _progress_lines(capsys)[-1]
    assert (last["step"], last["state"], last["detail"]) == ("clips", "failed", "cancelled")
