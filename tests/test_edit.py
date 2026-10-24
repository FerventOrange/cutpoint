"""``cutpoint edit``: analyze -> clips -> assemble in one command (#2)."""

from __future__ import annotations

import logging

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
    """A source video and the heavy steps replaced by fakes that record their calls."""
    video = tmp_path / "videos" / "My Clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"source bytes")
    calls: list[tuple[str, tuple, dict]] = []

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
        final = clips_dir.parent / "final.mp4"
        final.write_bytes(b"final video")
        final.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")
        return final

    monkeypatch.setattr(cutpoint.analyze, "analyze", analyze)
    monkeypatch.setattr(edit, "export_clips", export_clips)
    monkeypatch.setattr(edit, "assemble", assemble)
    return video, calls


def test_edit_runs_the_steps_in_order_and_names_the_final(pipeline, home):
    video, calls = pipeline
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
    video, calls = pipeline
    clips_dir = home / "projects" / "My Clip" / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "001_old.mp4").write_bytes(b"old")
    assert cli.main(["edit", str(video)]) == 1
    assert calls == []
    assert f"{clips_dir} is not empty; use --force to replace its clips" in caplog.text
    assert cli.main(["edit", str(video), "--force"]) == 0
    assert calls[1][2]["force"] is True


def test_edit_refuses_the_source_folder_as_project(pipeline, caplog):
    video, calls = pipeline
    caplog.set_level(logging.ERROR)
    assert cli.main(["edit", str(video), "--out", str(video.parent)]) == 1
    assert calls == []
    assert sorted(p.name for p in video.parent.iterdir()) == ["My Clip.mp4"]
