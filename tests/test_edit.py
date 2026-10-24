"""``cutpoint edit``: analyze -> clips -> assemble in one command (#2)."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import cutpoint.analyze
from cutpoint import cli, edit


@pytest.mark.parametrize(
    ("kind", "overrides", "expected"),
    [
        ("highlights", {}, ("action", None, 180, "chronological", "fade", False)),
        ("funny-moments", {}, ("funny", 8, None, "chronological", "cut", True)),
        ("montage", {"style": "gaming", "top": 5}, ("gaming", 5, 60, "score", "fade", False)),
    ],
)
def test_kind_table_and_overrides(kind, overrides, expected):
    k = edit.resolve_kind(kind, **overrides)
    assert (k.style, k.top, k.max_total, k.order, k.transition, k.captions) == expected
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
    }  # fmt: skip
    assert calls[2][1] == (project / "clips",)
    assert calls[2][2] == {"transition": "cut", "loudnorm": True, "captions": True}
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


def test_edit_name_of_a_drive_root_is_edit(tmp_path):
    """The rule itself is utils.input_name (tests/test_utils.py); edit's fallback is "edit"."""
    assert edit.edit_name(Path(tmp_path.anchor)) == "edit"


@pytest.mark.parametrize(
    ("argv", "vision"), [([], None), (["--no-vision"], False), (["--vision"], True)]
)
def test_edit_passes_vision_to_analyze(pipeline, argv, vision):
    """No option: analyze's own default (the style's, else Ollama if reachable)."""
    assert cli.main(["edit", str(pipeline.video), *argv]) == 0
    assert pipeline.calls[0][2] == {"vision": vision}


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
