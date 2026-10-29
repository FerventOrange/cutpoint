"""``assemble --format`` and ``assemble <project>``: timelines only on request, and the final
video from temporary clips that are removed afterwards (#519)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import envelope, needs_ffmpeg
from cutpoint import assemble, cli, clips, ffmpeg_utils, sources, srt
from cutpoint.ffmpeg_utils import staging_path
from cutpoint.pano import stitch


def _words(_src, _lang=None):
    return [{"word": f"w{n}", "start": n * 0.5, "end": n * 0.5 + 0.4} for n in range(20)]


def _clip(out: Path, seconds: float = 2.0) -> None:
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=size=64x48:rate=25:duration={seconds}", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={seconds}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )  # fmt: skip


@pytest.fixture
def clips_dir(tmp_path, monkeypatch) -> Path:
    """Two 2 s clips with a manifest whose source has a (stubbed) transcript."""
    monkeypatch.setattr(sources, "source_words", _words)
    folder = tmp_path / "proj" / "clips"
    folder.mkdir(parents=True)
    entries = []
    for n in (1, 2):
        _clip(folder / f"00{n}_m.mp4")
        entries.append({"file": f"00{n}_m.mp4", "source": str(tmp_path / "src.mp4"),
                        "start": 2.0 * n, "end": 2.0 * n + 2, "audio_fade": 0.25})  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    return folder


def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    code = cli.main([*argv, "--json"])
    return code, envelope(capsys.readouterr().out)


# ------------------------------------------------------------------- assemble <clips_dir>
@needs_ffmpeg
def test_mp4_writes_no_timelines(clips_dir, capsys):
    code, doc = _json(capsys, ["assemble", str(clips_dir)])
    project = clips_dir.parent
    assert code == 0
    assert set(doc["outputs"]) == {"format", "final", "srt", "duration", "clip_count"}
    assert doc["outputs"]["format"] == "mp4"
    assert sorted(p.name for p in project.iterdir()) == ["clips", "final.mp4", "final.srt"]


@needs_ffmpeg
@pytest.mark.parametrize("fmt", ["fcpxml", "otio"])
@pytest.mark.parametrize("transition", ["cut", "fade"])
def test_timeline_format_writes_only_the_timeline_and_srt_without_encoding(
    clips_dir, capsys, monkeypatch, fmt, transition
):
    def no_encode(*a, **k):
        raise AssertionError("nothing may be joined or encoded")

    for name in ("_join", "concat_copy", "concat_reencode", "crossfade", "crossfade_copy"):
        monkeypatch.setattr(assemble, name, no_encode)
    monkeypatch.setattr(assemble, "encode_with_fallback", no_encode)
    argv = ["assemble", str(clips_dir), "--format", fmt, "--transition", transition]
    code, doc = _json(capsys, argv)
    project = clips_dir.parent
    assert code == 0, doc
    out = doc["outputs"]
    assert set(out) == {"format", fmt, "srt", "duration", "clip_count"}
    assert out["format"] == fmt and out[fmt] == str(project / f"final.{fmt}")
    assert out["srt"] == str(project / "final.srt") and out["clip_count"] == 2
    fade = 0.24 if transition == "fade" else 0.0  # 6 frames at 25 fps
    infos = [assemble.probe(f) for f in sorted(clips_dir.glob("*.mp4"))]
    first = infos[0].vduration - fade if fade else infos[0].duration  # as in final.mp4
    assert out["duration"] == pytest.approx(first + infos[1].vduration, abs=0.001)
    assert sorted(p.name for p in project.iterdir()) == sorted(
        ["clips", "final.srt", f"final.{fmt}"]
    )
    assert (clips_dir / "001_m.mp4").as_uri() in (project / f"final.{fmt}").read_text("utf-8")


@needs_ffmpeg
def test_timeline_format_follows_out_and_the_mp4_is_left_as_it_was(clips_dir, caplog):
    project = clips_dir.parent
    (project / "cut.mp4").write_bytes(b"an earlier final")
    assert cli.main(["assemble", str(clips_dir), "--out", str(project / "cut.mp4"),
                     "--format", "otio"]) == 0  # fmt: skip
    assert (project / "cut.otio").is_file() and (project / "cut.srt").is_file()
    assert (project / "cut.mp4").read_bytes() == b"an earlier final"
    assert f"{project / 'cut.mp4'} is from an earlier run" in caplog.text


@pytest.mark.parametrize("flag", ["--loudnorm", "--captions"])
def test_timeline_format_refuses_options_of_the_final_video(tmp_path, capsys, flag):
    with pytest.raises(SystemExit) as exc:
        cli.main(["assemble", str(tmp_path), "--format", "fcpxml", flag])
    assert exc.value.code == 2
    assert f"{flag} applies to the final video" in capsys.readouterr().err


def test_project_options_need_a_project(tmp_path, capsys):
    code, doc = _json(capsys, ["assemble", str(tmp_path), "--top", "3"])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert doc["error"]["message"].startswith("--top needs a project")


# --------------------------------------------------------------------- assemble <project>
@pytest.fixture
def project(video, home, monkeypatch) -> Path:
    """A project with highlights.json of two moments of ``video`` (transcript stubbed)."""
    monkeypatch.setattr(sources, "source_words", _words)
    folder = home / "projects" / "p"
    folder.mkdir(parents=True)
    cands = [
        {"id": f"h00{n}", "source": str(video), "start": s, "end": s + 2.0, "score": 90 - n,
         "label": f"m{n}", "transcript": ""}
        for n, s in ((1, 1.0), (2, 6.0))
    ]  # fmt: skip
    hl = {"style": "action", "candidates": cands, "suggested_top": 2}
    (folder / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    return folder


NO_SHAPING = ["--no-speech-aware", "--no-smart-tail", "--no-event-tails"]


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


@needs_ffmpeg
def test_project_mp4_leaves_only_the_final_and_never_touches_clips(project, capsys):
    clips = project / "clips"
    clips.mkdir()
    (clips / "001_old.mp4").write_bytes(b"earlier clip")
    code, doc = _json(capsys, ["assemble", "p", "--top", "2", *NO_SHAPING])
    assert code == 0, doc
    assert doc["project"] == str(project)
    assert set(doc["outputs"]) == {"format", "final", "srt", "duration", "clip_count"}
    assert doc["outputs"]["clip_count"] == 2
    assert [w["code"] for w in doc["warnings"]] == ["temporary_clips"]
    assert _names(project) == ["clips", "final.mp4", "final.srt", "highlights.json"]
    assert _names(clips) == ["001_old.mp4"]
    assert (clips / "001_old.mp4").read_bytes() == b"earlier clip"
    assert assemble.probe(project / "final.mp4").duration == pytest.approx(4.0, abs=0.3)


@needs_ffmpeg
def test_project_segments_vertical_names_the_final(video, home, tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "source_words", _words)
    segments = tmp_path / "seg.json"
    segments.write_text(json.dumps([{"source": str(video), "start": 1, "end": 3}]), "utf-8")
    argv = ["assemble", "new", "--segments", str(segments), "--vertical", *NO_SHAPING]
    assert cli.main(argv) == 0
    folder = home / "projects" / "new"
    assert _names(folder) == ["final_vertical.mp4", "final_vertical.srt"]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("error", "code"), [(RuntimeError("join failed"), 1), (KeyboardInterrupt(), 130)]
)
def test_project_failure_or_interrupt_leaves_no_temporary_folder(
    project, capsys, monkeypatch, error, code
):
    seen = []

    def join(files, *a, **k):
        seen.append(files[0].parent)  # the temporary clips, exported by now
        raise error

    monkeypatch.setattr(assemble, "_join", join)
    got, doc = _json(capsys, ["assemble", "p", *NO_SHAPING])
    assert got == code
    assert doc["error"]["code"] == ("interrupted" if code == 130 else "error")
    assert seen and seen[0].name.startswith(".assemble-tmp-") and seen[0].parent == project
    assert _names(project) == ["highlights.json"]


@needs_ffmpeg
def test_project_with_too_little_space_stops_before_exporting(project, capsys, monkeypatch):
    monkeypatch.setattr(stitch, "free_gb", lambda folder: 0.5)
    monkeypatch.setattr(assemble, "_join", lambda *a, **k: pytest.fail("joined"))
    code, doc = _json(capsys, ["assemble", "p", *NO_SHAPING])
    assert code == 1
    assert (
        "0.5 GB free, the temporary clips and the final video need about"
        in (doc["error"]["message"])
    )
    assert _names(project) == ["highlights.json"]


def test_project_mp4_refuses_force(home, capsys):
    folder = home / "projects" / "p"
    folder.mkdir(parents=True)
    (folder / "highlights.json").write_text("{}", encoding="utf-8")
    code, doc = _json(capsys, ["assemble", "p", "--force"])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert "--force replaces kept clips" in doc["error"]["message"]
    assert _names(folder) == ["highlights.json"]


@needs_ffmpeg
def test_project_timeline_format_keeps_the_clips(project, capsys):
    code, doc = _json(capsys, ["assemble", "p", "--format", "fcpxml", *NO_SHAPING])
    assert code == 0, doc
    clip_keys = {"clips_dir", "manifest", "clips", "count", "total_seconds"}  # kept: listed
    assert set(doc["outputs"]) == {"format", "fcpxml", "srt", "duration", "clip_count"} | clip_keys
    assert doc["outputs"]["clips_dir"] == str(project / "clips")
    assert doc["warnings"] == []
    assert _names(project) == ["clips", "final.fcpxml", "final.srt", "highlights.json"]
    assert len(list((project / "clips").glob("*.mp4"))) == 2
    # A non-empty clips folder needs --force, as for clips
    code, doc = _json(capsys, ["assemble", "p", "--format", "otio", *NO_SHAPING])
    assert code == 1 and doc["error"]["code"] == "output_not_empty"
    code, doc = _json(capsys, ["assemble", "p", "--format", "otio", "--force", *NO_SHAPING])
    assert code == 0 and (project / "final.otio").is_file()


# ------------------------------------------------------- temporary clips: cleanup (review)
def test_temporary_clips_remove_the_360_paths_and_staging_folders(tmp_path):
    project = tmp_path / "p"
    with assemble.temporary_clips(project) as tmp:
        (tmp / "001_a.mp4").parent.mkdir()
        (tmp / "001_a.mp4").write_bytes(b"clip")
        staging_path(tmp).mkdir()
        paths = project / "pano_paths" / tmp.name  # PanoExporter.paths_dir of the export
        paths.mkdir(parents=True)
        (paths / "001_a.json").write_text("{}", encoding="utf-8")
    assert _names(project) == []  # pano_paths too, once empty
    with assemble.temporary_clips(project) as tmp:
        (project / "pano_paths" / "clips").mkdir(parents=True)  # a clips export's paths
        (project / "pano_paths" / tmp.name).mkdir()
    assert _names(project) == ["pano_paths"] and _names(project / "pano_paths") == ["clips"]


def test_temporary_clips_sweep_what_a_hard_killed_run_left(tmp_path):
    project = tmp_path / "p"
    old = project / ".assemble-tmp-old"
    old.mkdir(parents=True)
    (old / "001_a.mp4").write_bytes(b"clip")
    staging_path(old).mkdir()
    (project / "pano_paths" / old.name).mkdir(parents=True)
    (project / "..assemble-tmp-old.cutpoint-lock").write_text("{}", encoding="utf-8")
    (project / "final.mp4").write_bytes(b"kept")
    with assemble.temporary_clips(project):
        assert _names(project) == ["..assemble-tmp-" + _run_name(project), "final.mp4"]
    assert _names(project) == ["final.mp4"]


def _run_name(project: Path) -> str:
    """``<id>.run.cutpoint-lock`` of the one run lock in ``project``."""
    [lock] = [p.name for p in project.iterdir() if p.name.endswith(".run.cutpoint-lock")]
    return lock.removeprefix("..assemble-tmp-")


def test_temporary_clips_keep_the_folder_of_a_run_still_going(tmp_path):
    project = tmp_path / "p"
    busy = project / ".assemble-tmp-busy"
    busy.mkdir(parents=True)
    with ffmpeg_utils.swap_lock(project / ".assemble-tmp-busy.run"):  # the other run's lock
        with assemble.temporary_clips(project):
            pass
        assert busy.is_dir()
    with assemble.temporary_clips(project):  # gone with its run: swept
        pass
    assert _names(project) == []


def test_a_sweep_that_cannot_take_a_lock_does_not_fail_the_run(tmp_path, monkeypatch, caplog):
    project = tmp_path / "p"
    old = project / ".assemble-tmp-old"
    old.mkdir(parents=True)
    real = assemble.swap_lock

    def swap_lock(dst):
        if dst.name == ".assemble-tmp-old.run":
            raise PermissionError(13, "denied", str(dst))
        return real(dst)

    monkeypatch.setattr(assemble, "swap_lock", swap_lock)
    with assemble.temporary_clips(project) as tmp:
        tmp.mkdir()
    assert not tmp.exists() and old.is_dir()  # kept, with a WARNING
    assert f"Could not check {old} left by an earlier run" in caplog.text


@needs_ffmpeg
@pytest.mark.parametrize(("error", "code"), [(RuntimeError("bad"), 1), (KeyboardInterrupt(), 130)])
def test_project_failure_during_the_export_with_partial_clips_leaves_nothing(
    project, monkeypatch, error, code
):
    def render(plans, work, **kw):
        work.mkdir(exist_ok=True)
        (work / plans[0].name).write_bytes(b"half a clip")
        raise error

    monkeypatch.setattr(clips, "render_clips", render)
    assert cli.main(["assemble", "p", *NO_SHAPING]) == code
    assert _names(project) == ["highlights.json"]


@needs_ffmpeg
def test_project_failure_in_the_final_srt_leaves_no_temporary_clips(project, monkeypatch):
    def fail(*a, **k):
        raise RuntimeError("srt failed")

    monkeypatch.setattr(srt, "write_final_srt", fail)
    assert cli.main(["assemble", "p", *NO_SHAPING]) == 1
    assert not [n for n in _names(project) if "assemble-tmp" in n]


@needs_ffmpeg
def test_a_folder_that_cannot_be_removed_keeps_the_error_and_is_named(
    project, capsys, caplog, monkeypatch
):
    real = shutil.rmtree

    def rmtree(path, *a, **k):
        if Path(path).name.startswith(".assemble-tmp-"):
            raise PermissionError(13, "in use", str(path))
        return real(path, *a, **k)

    def join(*a, **k):
        raise RuntimeError("join failed")

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    monkeypatch.setattr(assemble, "_join", join)
    code, doc = _json(capsys, ["assemble", "p", *NO_SHAPING])
    assert code == 1 and doc["error"]["message"] == "join failed"
    [left] = [n for n in _names(project) if n.startswith(".assemble-tmp-")]
    assert f"Cannot remove the temporary clips {project / left}" in caplog.text
    assert [(w["code"], w["files"]) for w in doc["warnings"]] == [
        ("temporary_clips_left", [str(project / left)])
    ]


@needs_ffmpeg
def test_success_with_a_folder_left_does_not_say_removed(project, capsys, monkeypatch):
    real = shutil.rmtree

    def rmtree(path, *a, **k):
        if Path(path).name.startswith(".assemble-tmp-"):
            raise OSError(16, "busy", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    code, doc = _json(capsys, ["assemble", "p", *NO_SHAPING])
    assert code == 0
    assert [w["code"] for w in doc["warnings"]] == ["temporary_clips_left"]


# ------------------------------------------------------------ clips folder or project (F4, F5)
def test_segments_with_a_clips_folder_is_refused(tmp_path, capsys):
    folder = tmp_path / "clips"
    folder.mkdir()
    (folder / "manifest.json").write_text('{"clips": []}', encoding="utf-8")
    code, doc = _json(capsys, ["assemble", str(folder), "--segments", str(tmp_path / "s.json")])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert doc["error"]["message"] == (
        f"{folder} is a clips folder; --segments needs the project folder"
    )


@pytest.mark.parametrize("mark", ["clips", "clips_vertical", "project.json"])
def test_a_project_without_highlights_is_refused(tmp_path, capsys, mark):
    folder = tmp_path / "p"
    folder.mkdir()
    if mark.endswith(".json"):
        (folder / mark).write_text("{}", encoding="utf-8")
    else:
        (folder / mark).mkdir()
    code, doc = _json(capsys, ["assemble", str(folder)])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert doc["error"]["message"] == (
        f"{folder} is a project without highlights.json: run analyze, pass --segments, or give "
        "its clips folder"
    )


@needs_ffmpeg
def test_clips_folders_with_a_manifest_still_assemble(clips_dir, capsys):
    """A clips folder (the UI's clips_ui_<n> too) holds manifest.json: never a project."""
    ui = clips_dir.rename(clips_dir.with_name("clips_ui_2"))
    (ui / "clips").mkdir()  # even with a sub-folder of that name
    (ui.parent / "project.json").write_text("{}", encoding="utf-8")
    code, doc = _json(capsys, ["assemble", str(ui), "--format", "otio"])
    assert code == 0 and doc["outputs"]["otio"] == str(ui.parent / "final_ui_2.otio")
