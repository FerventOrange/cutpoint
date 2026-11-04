"""#109 phase 6: an interrupted clips or extend export keeps its finished clips in the hidden
``.<folder>.cutpoint-resume`` and the next run with the same plan encodes only the rest."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from conftest import media_info
from cutpoint import cache, cli, clips, extend, ffmpeg_utils, folders, planning, runstate
from cutpoint.utils import clip_files

NAMES = ["001_a.mp4", "002_b.mp4", "003_c.mp4", "004_d.mp4"]


@pytest.fixture(autouse=True)
def cpu_encoder(monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    monkeypatch.setattr(clips, "probe", media_info)
    monkeypatch.setattr(extend, "probe", media_info)
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))


def _project(home: Path, shift: float = 0.0) -> Path:
    project = home / "projects" / "p"
    project.mkdir(parents=True, exist_ok=True)
    src = project / "src.mp4"
    src.write_bytes(b"x")
    segs = [
        {"source": str(src), "start": i * 5.0 + shift, "end": i * 5.0 + 2 + shift, "label": n[4]}
        for i, n in enumerate(NAMES)
    ]
    (project / "seg.json").write_text(json.dumps(segs), encoding="utf-8")
    return project


class _Encoder:
    """Fake export_clip_checked: records the clips it encodes; ``stop_at`` (a file name)
    raises Ctrl+C before (``during=False``) or half-way through (a partial file) that clip."""

    def __init__(self, monkeypatch, stop_at: str | None = None, during: bool = False) -> None:
        self.encoded: list[str] = []
        self.stop_at, self.during = stop_at, during
        monkeypatch.setattr(clips, "export_clip_checked", self)
        monkeypatch.setattr(extend, "export_clip_checked", self)

    def __call__(self, src, info, start, end, out, *_a, **_kw):
        if out.name == self.stop_at:
            if self.during:
                folders.part_path(out).write_bytes(b"half")
            raise KeyboardInterrupt
        self.encoded.append(out.name)
        out.write_bytes(f"clip {out.name} {start:.2f}".encode())


def _clips(project: Path, *extra: str) -> int:
    return cli.main(["clips", "p", "--segments", str(project / "seg.json"), *extra])


def _resume_dir(project: Path, name: str = "clips") -> Path:
    return project / f".{name}.cutpoint-resume"


@pytest.mark.parametrize("during", [False, True], ids=["after-clip-k", "during-clip-k+1"])
def test_interrupted_clips_resume_encodes_only_the_missing_ones(home, monkeypatch, during):
    project = _project(home)
    first = _Encoder(monkeypatch, stop_at="003_c.mp4", during=during)
    assert _clips(project) == 130
    assert first.encoded == ["001_a.mp4", "002_b.mp4"]
    kept = _resume_dir(project)
    assert not (project / "clips").exists()  # nothing for assemble to see
    assert not (project / ".clips.cutpoint-tmp").exists()
    done = json.loads((kept / "resume.json").read_text(encoding="utf-8"))["clips"]
    assert sorted(done) == ["001_a.mp4", "002_b.mp4"]  # never the partial one

    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == ["003_c.mp4", "004_d.mp4"]
    assert [p.name for p in clip_files(project / "clips")] == NAMES
    assert (project / "clips" / "001_a.mp4").read_bytes() == b"clip 001_a.mp4 0.00"
    assert not kept.exists()
    names = {p.name for p in (project / "clips").iterdir()}
    assert "resume.json" not in names and not any(n.startswith(".") for n in names)


def test_an_interrupted_force_run_leaves_the_old_clips_for_assemble(home, monkeypatch):
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    before = {p.name: p.read_bytes() for p in (project / "clips").iterdir()}
    _project(home, shift=1.0)
    _Encoder(monkeypatch, stop_at="002_b.mp4", during=True)
    assert _clips(project, "--force") == 130
    # assemble lists the clips folder: still the old set, nothing partial
    assert {p.name: p.read_bytes() for p in (project / "clips").iterdir()} == before
    assert sorted(p.name for p in _resume_dir(project).iterdir()) == [
        ".002_b.part.mp4",
        "001_a.mp4",
        "resume.json",
    ]
    folders.recover_swap(project / "clips")  # assemble's recovery leaves the resume folder
    assert _resume_dir(project).is_dir()

    again = _Encoder(monkeypatch)
    assert _clips(project, "--force") == 0
    assert again.encoded == NAMES[1:]
    assert not _resume_dir(project).exists()


def test_a_hard_kill_leftover_staging_folder_is_taken_over(home, monkeypatch):
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    # a hard kill: the staging folder was never renamed, with a partial clip in it
    tmp = project / ".clips.cutpoint-tmp"
    _resume_dir(project).rename(tmp)
    folders.part_path(tmp / "003_c.mp4").write_bytes(b"half")

    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == ["003_c.mp4", "004_d.mp4"]
    assert not tmp.exists() and not _resume_dir(project).exists()
    assert [p.name for p in clip_files(project / "clips")] == NAMES


def test_a_run_with_another_plan_deletes_the_resume_folder(home, monkeypatch):
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    assert _resume_dir(project).is_dir()
    _project(home, shift=0.5)  # other in and out points
    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == NAMES
    assert not _resume_dir(project).exists()
    assert (project / "clips" / "001_a.mp4").read_bytes() == b"clip 001_a.mp4 0.50"


def test_a_changed_resume_record_is_not_trusted(home, monkeypatch):
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    (_resume_dir(project) / "002_b.mp4").write_bytes(b"truncated")  # another size
    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == NAMES[1:]


def test_cutpoint_resume_of_a_recorded_clips_force(home, monkeypatch):
    project = _project(home)
    (project / "highlights.json").write_text(
        json.dumps({"style": "gaming", "candidates": []}), encoding="utf-8"
    )
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    _project(home, shift=1.0)
    _Encoder(monkeypatch, stop_at="003_c.mp4", during=True)
    assert _clips(project, "--force") == 130
    rec = runstate.read(project)
    assert rec["command"] == "clips" and rec["status"] == "interrupted"
    assert "--force" in rec["argv"]

    again = _Encoder(monkeypatch)
    assert cli.main(["resume", "p"]) == 0
    assert again.encoded == ["003_c.mp4", "004_d.mp4"]
    assert (project / "clips" / "004_d.mp4").read_bytes() == b"clip 004_d.mp4 16.00"
    assert runstate.read(project)["status"] == "done"
    assert not _resume_dir(project).exists()


def test_interrupted_extend_in_place_resumes(home, monkeypatch):
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    before = {p.name: p.read_bytes() for p in (project / "clips").iterdir()}
    clips_dir = str(project / "clips")
    argv = ["extend", clips_dir, "--seconds", "1", "--no-speech-aware", "--out", clips_dir]
    _Encoder(monkeypatch, stop_at="002_b.mp4")
    assert cli.main([*argv, "--force"]) == 130
    assert {p.name: p.read_bytes() for p in (project / "clips").iterdir()} == before
    assert _resume_dir(project).is_dir()

    again = _Encoder(monkeypatch)
    assert cli.main([*argv, "--force"]) == 0
    assert again.encoded == NAMES[1:]
    assert not _resume_dir(project).exists()


@pytest.mark.parametrize("out", ["new", "empty", "nested"])
def test_interrupted_extend_into_a_new_or_empty_out_folder_resumes(home, monkeypatch, out):
    """#721: as clips, extend into a new or empty --out keeps its finished clips in the hidden
    resume folder next to it, shows nothing partial and the next equal run takes them over."""
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    rel = {"new": "clips_x", "empty": "clips_x", "nested": "more/clips_x"}[out]
    out_dir = project / rel
    if out == "empty":
        out_dir.mkdir()
    argv = ["extend", str(project / "clips"), "--seconds", "1", "--no-speech-aware"]
    argv += ["--out", str(out_dir)]
    first = _Encoder(monkeypatch, stop_at="003_c.mp4", during=True)
    assert cli.main(argv) == 130
    assert first.encoded == NAMES[:2]
    kept = out_dir.parent / ".clips_x.cutpoint-resume"
    done = json.loads((kept / "resume.json").read_text(encoding="utf-8"))["clips"]
    assert sorted(done) == NAMES[:2]
    # nothing for assemble to see: no folder, or the empty one as it was
    assert (list(out_dir.iterdir()) == []) if out == "empty" else not out_dir.exists()
    assert not (out_dir.parent / ".clips_x.cutpoint-tmp").exists()

    again = _Encoder(monkeypatch)
    assert cli.main(argv) == 0
    assert again.encoded == NAMES[2:]
    assert [p.name for p in clip_files(out_dir)] == NAMES
    assert not kept.exists()
    names = {p.name for p in out_dir.iterdir()}
    assert "resume.json" not in names and not any(n.startswith(".") for n in names)


def test_extend_failing_before_any_clip_leaves_no_new_folder(home, monkeypatch):
    """#721: a new --out folder (and a parent it made) is not left behind empty."""
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    out_dir = project / "more" / "clips_x"
    argv = ["extend", str(project / "clips"), "--seconds", "1", "--no-speech-aware"]
    _Encoder(monkeypatch, stop_at="001_a.mp4")
    assert cli.main([*argv, "--out", str(out_dir)]) == 130
    assert not (project / "more").exists()


# -- review cycle 1 ---------------------------------------------------------------------------


def test_clips_runs_keep_an_interrupted_analysis_record(home, monkeypatch, caplog):
    """F1: neither a finished nor a failing clips run replaces the progress.json of an
    interrupted events run; cutpoint resume still replays events."""
    project = _project(home)
    (project / "highlights.json").write_text(json.dumps({"style": "gaming"}), encoding="utf-8")
    rec = runstate.Recorder("events", ["events", "p"], project, {})
    rec.data.update({"status": "interrupted", "exit_code": 130})
    cache.save_json(project / runstate.FILE, rec.data)
    before = (project / runstate.FILE).read_bytes()
    _Encoder(monkeypatch)
    with caplog.at_level(logging.INFO):
        assert _clips(project) == 0
    assert "progress.json keeps the interrupted 'events' run" in caplog.text
    assert _clips(project) == 1  # output_not_empty
    assert (project / runstate.FILE).read_bytes() == before
    replayed = []
    monkeypatch.setattr(cli, "_cmd_events", lambda a: replayed.append(str(a.project)))
    assert cli.main(["resume", "p"]) == 0
    assert replayed == ["p"]


def _edit_project(home: Path, monkeypatch) -> tuple[Path, Path]:
    """Project ``p`` analysed already (analyze is a no-op fake) with four candidates."""
    import cutpoint.analyze  # noqa: PLC0415
    from cutpoint import edit  # noqa: PLC0415

    project = _project(home)
    video = home / "videos" / "v.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"source")
    cands = [
        {"id": f"h00{i + 1}", "source": str(video), "start": i * 5.0, "end": i * 5.0 + 2,
         "duration": 2.0, "score": 90 - i, "label": n[4]}
        for i, n in enumerate(NAMES)
    ]  # fmt: skip
    highlights = {"style": "gaming", "candidates": cands, "suggested_top": 4}
    (project / "highlights.json").write_text(json.dumps(highlights), encoding="utf-8")
    monkeypatch.setattr(cutpoint.analyze, "analyze", lambda *a, **kw: None)
    monkeypatch.setattr(cutpoint.analyze, "report_highlights", lambda *a, **kw: None)

    def assemble(clips_dir, out=None, **kw):
        out = out or clips_dir.parent / "final.mp4"
        out.write_bytes(b"final")
        return out

    monkeypatch.setattr(edit, "assemble", assemble)
    return project, video


def test_cutpoint_resume_of_an_edit_interrupted_in_its_clips_step(home, monkeypatch):
    project, video = _edit_project(home, monkeypatch)
    argv = ["edit", str(video), "--out", "p", "--kind", "clips", "--top", "4"]
    _Encoder(monkeypatch, stop_at="003_c.mp4", during=True)
    assert cli.main(argv) == 130
    assert runstate.read(project)["status"] == "interrupted"
    again = _Encoder(monkeypatch)
    assert cli.main(["resume", "p"]) == 0
    assert again.encoded == ["003_c.mp4", "004_d.mp4"]
    assert [p.name for p in clip_files(project / "clips")] == NAMES
    assert not _resume_dir(project).exists()


def _resume_folders(home: Path) -> list[Path]:
    return [p for p in home.rglob("*") if p.name.endswith(".cutpoint-resume")]


def test_temporary_clips_keep_no_resume_folder(home, monkeypatch):
    project, video = _edit_project(home, monkeypatch)
    from cutpoint import assemble  # noqa: PLC0415

    monkeypatch.setattr(assemble, "_assemble", lambda *a, **kw: project / "final.mp4")
    _Encoder(monkeypatch, stop_at="003_c.mp4", during=True)
    assert cli.main(["assemble", "p", "--segments", str(project / "seg.json")]) == 130
    assert _resume_folders(home) == []
    edit_argv = ["edit", str(video), "--out", "p", "--top", "4", "--no-keep-clips"]
    assert cli.main(edit_argv) == 130
    assert _resume_folders(home) == []


def test_a_takeover_with_two_export_workers(home, monkeypatch):
    from cutpoint.encode import VideoEncoder  # noqa: PLC0415

    monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", "2")
    monkeypatch.setattr(clips, "video_encoder", lambda: VideoEncoder("nvenc"))
    project = _project(home)
    first = _Encoder(monkeypatch, stop_at="004_d.mp4")
    assert _clips(project) == 130
    done = json.loads((_resume_dir(project) / "resume.json").read_text(encoding="utf-8"))
    assert sorted(done["clips"]) == sorted(first.encoded)  # each recorded as it finished
    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert sorted(second.encoded) == sorted(set(NAMES) - set(first.encoded))
    assert [p.name for p in clip_files(project / "clips")] == NAMES
    assert not _resume_dir(project).exists()


def test_a_clip_with_a_read_error_is_exported_again(home, monkeypatch):
    """F6: as extend's _reusable, a clip trimmed by a read error is not taken over."""
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    path = _resume_dir(project) / "resume.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clips"]["001_a.mp4"]["read_error"] = {"end": 1.0}
    path.write_text(json.dumps(data), encoding="utf-8")
    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == ["001_a.mp4", "003_c.mp4", "004_d.mp4"]


def test_a_locked_staging_folder_keeps_its_finished_clips(home, monkeypatch):
    """F4: a rename that fails right after Ctrl+C never deletes the finished clips."""
    project = _project(home)
    real = Path.replace

    def replace(self, target):
        if Path(target).name == ".clips.cutpoint-resume":
            raise PermissionError(13, "locked")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    tmp = project / ".clips.cutpoint-tmp"
    assert sorted(p.name for p in tmp.iterdir()) == ["001_a.mp4", "002_b.mp4", "resume.json"]
    monkeypatch.setattr(Path, "replace", real)
    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == ["003_c.mp4", "004_d.mp4"]


def test_the_key_changes_with_the_lut_file_and_the_version(tmp_path, monkeypatch):
    """F7"""
    from cutpoint import clipresume  # noqa: PLC0415
    from cutpoint.encode import Lut, OutputFormat  # noqa: PLC0415

    src, cube = tmp_path / "a.mp4", tmp_path / "look.cube"
    src.write_bytes(b"x")
    cube.write_text("LUT_3D_SIZE 2\n", encoding="utf-8")
    fmt = OutputFormat(1920, 1080, 30)

    def key() -> str:
        return clipresume.clip_key(src, 0, 2, fmt, False, None, 0.25, "", Lut(cube))

    first = key()
    cube.write_text("LUT_3D_SIZE 33\n", encoding="utf-8")
    assert key() != first
    second = key()
    monkeypatch.setattr(clipresume, "__version__", "0.0.0-other")
    assert key() != second


def test_a_hard_kill_leftover_that_cannot_be_renamed_is_left_alone(home, monkeypatch):
    """F9: recover_swap stops with an error and never deletes finished clips it cannot keep;
    an older resume folder is removed only once the rename went ahead."""
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    older = _resume_dir(project)
    tmp = project / ".clips.cutpoint-tmp"
    shutil.copytree(older, tmp)  # a hard kill of a later run, next to an older resume folder
    real = Path.replace

    def replace(self, target):
        if self == tmp:
            raise PermissionError(13, "locked")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError, match="Cannot keep the finished clips"):
        folders.recover_swap(project / "clips")
    assert sorted(p.name for p in tmp.iterdir()) == ["001_a.mp4", "002_b.mp4", "resume.json"]
    assert older.is_dir()  # not removed before the rename
    monkeypatch.setattr(Path, "replace", real)
    folders.recover_swap(project / "clips")
    assert not tmp.exists() and older.is_dir()


@pytest.mark.parametrize("older", [False, True], ids=["alone", "over-an-older-one"])
def test_recover_swap_promotes_a_resume_new_folder_of_a_hard_kill(home, monkeypatch, older):
    """The newer clips of -new are kept; an older resume folder goes."""
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    keep = _resume_dir(project)
    new = keep.with_name(f"{keep.name}-new")
    if older:
        shutil.copytree(keep, new)
        (keep / "001_a.mp4").write_bytes(b"older")
    else:
        keep.rename(new)
    folders.recover_swap(project / "clips")
    assert not new.exists()
    assert sorted(p.name for p in keep.iterdir()) == ["001_a.mp4", "002_b.mp4", "resume.json"]
    assert (keep / "001_a.mp4").read_bytes() == b"clip 001_a.mp4 0.00"  # the newer one
    second = _Encoder(monkeypatch)
    assert _clips(project) == 0
    assert second.encoded == ["003_c.mp4", "004_d.mp4"]


def test_a_resume_new_folder_that_cannot_be_promoted_leaves_both(home, monkeypatch, caplog):
    project = _project(home)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project) == 130
    keep = _resume_dir(project)
    new = keep.with_name(f"{keep.name}-new")
    shutil.copytree(keep, new)
    real = shutil.rmtree

    def rmtree(path, *a, **kw):
        if Path(path) == keep:
            raise PermissionError(13, "locked")
        return real(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    with caplog.at_level(logging.WARNING):
        folders.recover_swap(project / "clips")
    assert keep.is_dir() and new.is_dir()
    assert "Could not tidy up" in caplog.text


@pytest.mark.parametrize("where", ["backing-up", "moving-in"])
def test_a_failed_swap_after_a_takeover_loses_no_new_clip(home, monkeypatch, where):
    """Windows F1: an old clip held open during the swap; the new clips stay resumable."""
    monkeypatch.setattr(cache.time, "sleep", lambda _s: None)
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    clips_dir = project / "clips"
    before = {p.name: p.read_bytes() for p in clips_dir.iterdir()}
    _project(home, shift=1.0)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _clips(project, "--force") == 130
    tmp = project / ".clips.cutpoint-tmp"
    real = Path.replace

    def replace(self, target):
        old = self == clips_dir / "003_c.mp4"  # the old clip, to the backup
        new = self == tmp / "003_c.mp4"  # the new one, into clips
        if old if where == "backing-up" else new:
            raise PermissionError(13, "locked")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    second = _Encoder(monkeypatch)
    assert _clips(project, "--force") == 1
    assert second.encoded == ["003_c.mp4", "004_d.mp4"]
    assert {p.name: p.read_bytes() for p in clips_dir.iterdir()} == before
    done = json.loads((_resume_dir(project) / "resume.json").read_text(encoding="utf-8"))
    assert sorted(done["clips"]) == NAMES
    monkeypatch.setattr(Path, "replace", real)
    third = _Encoder(monkeypatch)
    assert _clips(project, "--force") == 0
    assert third.encoded == []
    assert (clips_dir / "004_d.mp4").read_bytes() == b"clip 004_d.mp4 16.00"
    assert not _resume_dir(project).exists() and not tmp.exists()


def test_a_locked_staging_folder_warning_names_it_once(home, monkeypatch, caplog):
    """Windows F2: the WARNING names the staging folder and gives the hint once."""
    monkeypatch.setattr(cache.time, "sleep", lambda _s: None)
    project = _project(home)
    tmp = project / ".clips.cutpoint-tmp"
    real = Path.replace

    def replace(self, target):
        if self == tmp:
            err = PermissionError(13, "The process cannot access the file")
            err.winerror = 32
            raise err
        return real(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    with caplog.at_level(logging.WARNING):
        assert _clips(project) == 130
    warning = next(m for m in caplog.messages if m.startswith("Kept the finished clips"))
    assert str(tmp) in warning and "cutpoint-resume-new" not in warning
    assert "re-run" not in warning and "open in another program" in warning


def test_recover_swap_after_a_hard_kill_mid_swap_keeps_the_moved_in_clips(home, monkeypatch):
    """W1: killed while moving the new clips in (001/002 in clips, 003/004 still staged, the
    originals in the backup with the journal): recovery puts the originals back and moves the
    new clips back into the resume folder, so the next run encodes nothing."""
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    clips_dir = project / "clips"
    before = {p.name: p.read_bytes() for p in clips_dir.iterdir()}
    _project(home, shift=1.0)
    _Encoder(monkeypatch, stop_at="nothing")
    # the new set, exported but not swapped in: build it by interrupting just before the swap
    real_swap = folders.swap_in
    monkeypatch.setattr(
        clips.folders, "swap_in", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    assert _clips(project, "--force") == 130
    monkeypatch.setattr(clips.folders, "swap_in", real_swap)
    keep = _resume_dir(project)
    tmp = project / ".clips.cutpoint-tmp"
    keep.rename(tmp)
    new = {n: (tmp / n).read_bytes() for n in NAMES}
    bak = project / ".clips.cutpoint-bak"
    bak.mkdir()
    olds = sorted(before)
    for n in olds:
        (clips_dir / n).replace(bak / n)
    for n in NAMES[:2]:
        (tmp / n).replace(clips_dir / n)
    names = sorted(p.name for p in tmp.iterdir() if p.name != "resume.json")
    journal = {"old": olds, "new": sorted({*names, *NAMES[:2]}), "created": False}
    (bak / ".cutpoint-swap.json").write_text(json.dumps(journal), encoding="utf-8")

    folders.recover_swap(clips_dir)
    assert {p.name: p.read_bytes() for p in clips_dir.iterdir()} == before
    assert not bak.exists() and not tmp.exists()
    assert {n: (keep / n).read_bytes() for n in NAMES} == new
    again = _Encoder(monkeypatch)
    assert _clips(project, "--force") == 0
    assert again.encoded == []
    assert {n: (clips_dir / n).read_bytes() for n in NAMES} == new


# -- --json taken_over (#109 UI) ---------------------------------------------------------------
def _taken_over(capsys, argv: list[str]) -> tuple[int, object]:
    capsys.readouterr()
    code = cli.main([*argv, "--json"])
    doc = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return code, doc["outputs"].get("taken_over")


def test_clips_json_reports_the_clips_taken_over(home, monkeypatch, capsys):
    project = _project(home)
    seg = ["clips", "p", "--segments", str(project / "seg.json")]
    _Encoder(monkeypatch, stop_at="003_c.mp4")
    assert _taken_over(capsys, seg) == (130, 0)  # nothing to take over yet
    _Encoder(monkeypatch)
    assert _taken_over(capsys, seg) == (0, 2)
    _Encoder(monkeypatch)
    assert _taken_over(capsys, [*seg, "--force"]) == (0, 0)


def test_extend_json_reports_the_clips_taken_over(home, monkeypatch, capsys):
    project = _project(home)
    _Encoder(monkeypatch)
    assert _clips(project) == 0
    clips_dir = str(project / "clips")
    argv = [
        "extend",
        clips_dir,
        "--seconds",
        "1",
        "--no-speech-aware",
        "--out",
        clips_dir,
        "--force",
    ]
    _Encoder(monkeypatch, stop_at="002_b.mp4")
    assert _taken_over(capsys, argv) == (130, 0)
    _Encoder(monkeypatch)
    assert _taken_over(capsys, argv) == (0, 1)
