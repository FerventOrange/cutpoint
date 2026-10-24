"""``--json`` on the editing commands: analyze, clips, assemble, edit and extend (#4, phase 2).

End to end on a short lavfi video; whisper is replaced by a stub transcriber, Ollama by
``--no-vision``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import cutpoint.analyze
from cutpoint import cli, edit
from cutpoint.analyze import report_highlights
from cutpoint.events import cli as events_cli
from cutpoint.events import pipeline
from test_jsonout import _envelope

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
)

SEGMENTS = [
    {
        "start": 3.0,
        "end": 4.5,
        "text": " ha ha that was great",
        "words": [
            {"start": 3.0, "end": 3.3, "word": " ha"},
            {"start": 3.3, "end": 3.6, "word": " ha"},
            {"start": 3.6, "end": 3.9, "word": " that"},
            {"start": 3.9, "end": 4.1, "word": " was"},
            {"start": 4.1, "end": 4.5, "word": " great"},
        ],
    }
]


class StubTranscriber:
    """No whisper: every source "says" SEGMENTS."""

    model_name, device = "stub", "cpu"

    def __init__(self, **_kw) -> None:
        pass

    def transcribe(self, _wav: Path) -> list[dict]:
        return SEGMENTS

    def close(self) -> None:
        pass


@pytest.fixture
def video(home, tmp_path, monkeypatch) -> Path:
    """A 10 s 160x120 video, quiet with two loud bursts; the transcriber stubbed."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", StubTranscriber)
    out = tmp_path / "videos" / "clip.mp4"
    out.parent.mkdir()
    loud = "volume=enable='between(t,2,4.5)+between(t,7,8.5)':volume=30"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=160x120:rate=10:duration=10", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=10", "-af", f"volume=0.02,{loud}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        check=True,
    )  # fmt: skip
    return out


def _json(capsys, argv: list[str]) -> tuple[int, dict, str]:
    """Run ``cutpoint <argv> --json``: exit code, the one envelope, stderr (the logs are in
    caplog)."""
    code = cli.main([*argv, "--json"])
    out, err = capsys.readouterr()
    return code, _envelope(out), err


def _analyze(video: Path, capsys) -> None:
    argv = ["analyze", str(video), "--style", "funny", "--out", "j", "--no-vision", "--no-scenes"]
    assert cli.main(argv) == 0
    capsys.readouterr()


def _clips(capsys, *extra: str) -> None:
    assert cli.main(["clips", "j", "--top", "2", *extra]) == 0
    capsys.readouterr()


def _assert_clips(outputs: dict, clips_dir: Path) -> None:
    assert outputs["clips_dir"] == str(clips_dir)
    assert outputs["manifest"] == str(clips_dir / "manifest.json")
    manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))["clips"]
    assert [c["file"] for c in outputs["clips"]] == [str(clips_dir / e["file"]) for e in manifest]
    assert outputs["count"] == len(manifest) > 0
    total = sum(e["end"] - e["start"] for e in manifest)
    assert outputs["total_seconds"] == pytest.approx(total, abs=0.001)
    for c, e in zip(outputs["clips"], manifest, strict=True):
        assert set(c) == {"file", "srt", "label", "source", "start", "end", "score"}
        assert Path(c["file"]).is_file()
        assert c["srt"] is None or Path(c["srt"]).is_file()
        assert (c["label"], c["source"], c["start"], c["end"]) == (
            e["label"], e["source"], e["start"], e["end"],
        )  # fmt: skip


# --------------------------------------------------------------------------------- analyze
@needs_ffmpeg
def test_analyze_json(video, home, capsys):
    code, doc, _ = _json(
        capsys,
        ["analyze", str(video), "--style", "funny", "--out", "j", "--no-vision", "--no-scenes"],
    )
    project = home / "projects" / "j"
    assert code == 0 and doc["ok"] is True and doc["command"] == "analyze"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert set(out) == {"highlights", "transcript", "candidates", "suggested_top", "style"}
    assert out["highlights"] == str(project / "highlights.json")
    assert out["transcript"] == str(project / "transcript.txt")
    assert all(Path(out[k]).is_file() for k in ("highlights", "transcript"))
    hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    assert out["candidates"] == len(hl["candidates"])
    assert out["suggested_top"] == hl["suggested_top"]
    assert out["style"] == "funny"
    assert doc["warnings"] == []


def test_analyze_json_missing_input(home, tmp_path, capsys):
    code, doc, _ = _json(capsys, ["analyze", str(tmp_path / "nope.mp4"), "--out", "j"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"]["code"] == "not_found"


# ----------------------------------------------------------------------------------- clips
@needs_ffmpeg
def test_clips_json(video, home, capsys, caplog):
    _analyze(video, capsys)
    code, doc, _ = _json(capsys, ["clips", "j", "--top", "2"])
    project = home / "projects" / "j"
    assert code == 0 and doc["ok"] is True and doc["command"] == "clips"
    assert doc["project"] == str(project)
    _assert_clips(doc["outputs"], project / "clips")
    assert doc["warnings"] == []

    # A non-empty clips folder without --force
    code, doc, _ = _json(capsys, ["clips", "j", "--top", "2"])
    assert code == 1 and doc["error"]["code"] == "output_not_empty"
    assert doc["error"]["message"] in caplog.text


@needs_ffmpeg
def test_clips_json_force_reports_kept_clips_and_a_blocked_name(video, home, capsys, caplog):
    _analyze(video, capsys)
    _clips(capsys)
    clips_dir = home / "projects" / "j" / "clips"
    first = sorted(clips_dir.glob("*.mp4"))[0]
    mine = first.rename(clips_dir / "099_mine.mp4")  # renamed during review

    code, doc, _ = _json(capsys, ["clips", "j", "--top", "2", "--force"])
    assert code == 0 and doc["ok"] is True
    [warning] = doc["warnings"]
    assert warning["code"] == "kept_clips"
    assert warning["files"] == [str(mine)]
    assert warning["message"].startswith(f"Kept 099_mine.mp4 in {clips_dir}: the old manifest")
    assert warning["message"] in caplog.text
    _assert_clips(doc["outputs"], clips_dir)

    # A clip the manifest does not list, under the name of a new clip, stops the run.
    manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["clips"] = manifest["clips"][1:]
    (clips_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, doc, _ = _json(capsys, ["clips", "j", "--top", "2", "--force"])
    assert code == 1 and doc["error"]["code"] == "output_not_empty"
    assert doc["error"]["message"].startswith("Cannot replace ")


def test_clips_json_missing_project(home, capsys):
    code, doc, _ = _json(capsys, ["clips", "nosuch", "--top", "2"])
    assert code == 1 and doc["error"]["code"] == "not_found"
    assert doc["project"] == str(home / "projects" / "nosuch")


def test_clips_json_bad_crop_is_bad_args_and_unchanged_without_json(home, capsys, caplog):
    code, doc, _ = _json(capsys, ["clips", "nosuch", "--crop", "wide"])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert doc["error"]["message"] == "Invalid crop 'wide'; use auto, none or W:H:X:Y"
    caplog.clear()
    assert cli.main(["clips", "nosuch", "--crop", "wide"]) == 1
    assert capsys.readouterr().out == ""
    assert "Invalid crop 'wide'; use auto, none or W:H:X:Y" in caplog.text


# -------------------------------------------------------------------------------- assemble
@needs_ffmpeg
def test_assemble_json(video, home, capsys):
    _analyze(video, capsys)
    _clips(capsys)
    project = home / "projects" / "j"
    code, doc, _ = _json(capsys, ["assemble", "projects/j/clips"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "assemble"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert set(out) == {"final", "fcpxml", "otio", "srt", "duration", "clip_count"}
    assert out["final"] == str(project / "final.mp4")
    assert out["fcpxml"] == str(project / "final.fcpxml")
    assert out["otio"] == str(project / "final.otio")
    assert out["srt"] == str(project / "final.srt")
    assert all(Path(out[k]).is_file() for k in ("final", "fcpxml", "otio", "srt"))
    assert out["duration"] > 0
    assert out["clip_count"] == len(list((project / "clips").glob("*.mp4")))


def test_assemble_json_missing_folder(home, capsys):
    code, doc, _ = _json(capsys, ["assemble", "projects/nosuch/clips"])
    assert code == 1 and doc["error"]["code"] == "not_found"
    assert doc["project"] == str(home / "projects" / "nosuch")


# ---------------------------------------------------------------------------------- extend
@needs_ffmpeg
def test_extend_json(video, home, capsys, caplog):
    _analyze(video, capsys)
    _clips(capsys)
    project = home / "projects" / "j"
    clips_dir = project / "clips"
    hand = clips_dir / "050_mine.mp4"
    shutil.copy2(sorted(clips_dir.glob("*.mp4"))[0], hand)  # added by hand: not re-exported
    argv = ["extend", "projects/j/clips", "--seconds", "0.5", "--no-speech-aware"]
    code, doc, _ = _json(capsys, argv)
    assert code == 0 and doc["ok"] is True and doc["command"] == "extend"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert set(out) == {"clips_dir", "manifest", "clips", "count", "total_seconds", "seconds"}
    _assert_clips(out, project / "clips_ext0p5s")
    assert out["seconds"] == {"start": 0.5, "end": 0.5}
    [warning] = doc["warnings"]
    assert warning["code"] == "skipped_clips" and warning["files"] == [str(hand)]
    assert "050_mine.mp4: no manifest entry" in caplog.text

    code, doc, _ = _json(capsys, argv)  # the output folder is not empty now
    assert code == 1 and doc["error"]["code"] == "output_not_empty"


@needs_ffmpeg
def test_extend_in_place_json_warns_kept_with_absolute_paths(
    video, home, capsys, monkeypatch, caplog
):
    _analyze(video, capsys)
    _clips(capsys)
    clips_dir = home / "projects" / "j" / "clips"
    hand = clips_dir / "050_mine.mp4"
    shutil.copy2(sorted(clips_dir.glob("*.mp4"))[0], hand)
    monkeypatch.chdir(clips_dir.parent)
    argv = ["extend", "clips", "--out", "clips", "--force", "--seconds", "0.2"]
    code, doc, _ = _json(capsys, [*argv, "--no-speech-aware"])
    assert code == 0 and doc["ok"] is True
    [warning] = doc["warnings"]
    assert warning["code"] == "kept_clips" and warning["files"] == [str(hand)]
    assert warning["message"].startswith(f"Kept 050_mine.mp4 in {clips_dir}: the old manifest")
    assert warning["message"] in caplog.text  # the absolute folder, not "clips" as typed
    assert "skipped" not in caplog.text
    assert hand.is_file()
    _assert_clips(doc["outputs"], clips_dir)  # "clips" as typed, made absolute


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--clip-override", "*", "extend"], "--clip-override * extend: expected KEY=SECONDS"),
        (["--clip-override", "*", "extend=x"], "--clip-override * extend=x: 'x' is not a number"),
        (["--no-speech-aware"], "Nothing to do: give --seconds (or --start / --end) > 0"),
    ],
)
def test_extend_argument_errors_are_bad_args(home, tmp_path, capsys, caplog, extra, message):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "manifest.json").write_text('{"clips": []}', encoding="utf-8")
    code, doc, _ = _json(capsys, ["extend", str(clips_dir), *extra])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert doc["error"]["message"].startswith(message) and message in caplog.text
    caplog.clear()
    assert cli.main(["extend", str(clips_dir), *extra]) == 1  # unchanged without --json
    assert capsys.readouterr().out == ""
    assert message in caplog.text


def test_extend_json_missing_folder(home, tmp_path, capsys):
    code, doc, _ = _json(capsys, ["extend", str(tmp_path / "nope"), "--seconds", "1"])
    assert code == 1 and doc["error"]["code"] == "not_found"


# ------------------------------------------------------------------------------------ edit
EDIT_KEYS = {
    "kind", "highlights", "transcript", "candidates", "suggested_top", "style",
    "clips_dir", "manifest", "clips", "count", "total_seconds",
    "final", "fcpxml", "otio", "srt", "duration", "clip_count", "copy", "copy_srt",
}  # fmt: skip


@needs_ffmpeg
def test_edit_json_is_one_envelope_with_the_sub_steps_outputs_and_warnings(
    video, home, capsys, caplog, monkeypatch
):
    # No loudnorm: its limiter resamples with soxr, which some ffmpeg builds lack.
    monkeypatch.setitem(edit.KINDS, "clips", replace(edit.KINDS["clips"], loudnorm=False))
    argv = ["edit", str(video), "--kind", "clips", "--top", "2", "--no-vision"]
    code, doc, _ = _json(capsys, argv)
    project = home / "projects" / "clip"
    assert code == 0 and doc["ok"] is True and doc["command"] == "edit"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert set(out) == EDIT_KEYS
    assert out["kind"] == "clips" and out["style"] == "action"
    assert out["copy"] == str(project / "clip-final.mp4")
    assert out["copy_srt"] == str(project / "clip-final.srt")
    assert out["final"] == str(project / "final.mp4")
    for key in ("highlights", "transcript", "manifest", "final", "fcpxml", "otio", "srt"):
        assert Path(out[key]).is_file(), key
    assert Path(out["copy"]).is_file() and Path(out["copy_srt"]).is_file()
    _assert_clips(out, project / "clips")
    assert doc["warnings"] == []

    # A non-empty clips folder without --force, before any analysis
    code, doc, _ = _json(capsys, argv)
    assert code == 1 and doc["error"]["code"] == "output_not_empty"

    # --force with a clip renamed during review: the clips step's warning is in the envelope.
    first = sorted((project / "clips").glob("*.mp4"))[0]
    mine = first.rename(project / "clips" / "099_mine.mp4")
    code, doc, _ = _json(capsys, [*argv, "--force"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "edit"
    assert [(w["code"], w["files"]) for w in doc["warnings"]] == [("kept_clips", [str(mine)])]
    assert doc["warnings"][0]["message"] in caplog.text
    assert doc["outputs"]["clip_count"] == doc["outputs"]["count"] + 1  # assembled with it


def test_edit_json_missing_input(home, tmp_path, capsys):
    code, doc, _ = _json(capsys, ["edit", str(tmp_path / "nope.mp4"), "--no-vision"])
    assert code == 1 and doc["error"]["code"] == "not_found"
    assert doc["command"] == "edit"


# ------------------------------------------------------------------------- without --json
@needs_ffmpeg
def test_without_json_stdout_stays_empty(video, home, capsys):
    """No command printed to stdout before --json; none does now (logs go to stderr)."""
    argv = ["analyze", str(video), "--style", "funny", "--out", "j", "--no-vision", "--no-scenes"]
    for args in (
        argv,
        ["clips", "j", "--top", "2"],
        ["assemble", "projects/j/clips"],
        ["extend", "projects/j/clips", "--seconds", "0.5", "--no-speech-aware"],
    ):
        assert cli.main(args) == 0, args
        assert capsys.readouterr().out == "", args
    assert cli.main(["clips", "j", "--top", "2"]) == 1  # not empty: same exit code as before
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------------ follow-ups
VISION_MSG = "--vision requested but Ollama/model unavailable; continuing without it"


class _NoOllama:
    """A VisionRater whose server or model is not there."""

    def __init__(self, **_kw) -> None:
        pass

    def available(self) -> bool:
        return False


@needs_ffmpeg
@pytest.mark.parametrize(
    ("flags", "style_off", "message"),
    [
        ([], False, "Ollama/model unavailable; continuing without vision"),
        (["--vision"], False, VISION_MSG),
        (["--no-vision"], False, None),
        ([], True, None),  # the style turns vision off by default (as ski does)
    ],
)
def test_analyze_json_warns_when_vision_was_wanted_but_unavailable(
    video, home, capsys, caplog, monkeypatch, flags, style_off, message
):
    monkeypatch.setattr(cutpoint.analyze, "VisionRater", _NoOllama)
    if style_off:
        funny = replace(cli.get_style("funny"), vision_default=False)
        monkeypatch.setattr(cli, "get_style", lambda name: funny)
    argv = ["analyze", str(video), "--style", "funny", "--out", "j", "--no-scenes", *flags]
    code, doc, _ = _json(capsys, argv)
    assert code == 0 and doc["ok"] is True and doc["outputs"]["candidates"] > 0
    expected = [] if message is None else [("vision_unavailable", message, [])]
    assert [(w["code"], w["message"], w["files"]) for w in doc["warnings"]] == expected
    # The log line is as before: only for an explicit --vision.
    assert (VISION_MSG in caplog.text) is (flags == ["--vision"])


def test_clips_json_unknown_ids_is_bad_args(home, capsys, caplog):
    project = home / "projects" / "j"
    project.mkdir(parents=True)
    cand = {"id": "h001", "source": "a.mp4", "start": 0, "end": 1, "score": 50}
    hl = {"style": "funny", "candidates": [cand]}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    code, doc, _ = _json(capsys, ["clips", "j", "--ids", "h001", "h009"])
    assert code == 1 and doc["error"] == {"code": "bad_args", "message": "Unknown ids: h009"}
    caplog.clear()
    assert cli.main(["clips", "j", "--ids", "h009"]) == 1  # unchanged without --json
    assert capsys.readouterr().out == "" and "Unknown ids: h009" in caplog.text


def test_edit_events_json_reports_the_re_ranked_highlights(home, tmp_path, capsys, monkeypatch):
    """events re-ranks highlights.json after analyze: the envelope has the later values."""
    video = tmp_path / "videos" / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"source bytes")

    def analyze(targets, style, out, **kw):
        out.mkdir(parents=True, exist_ok=True)
        (out / "transcript.txt").write_text("", encoding="utf-8")
        hl = {"style": style.name, "candidates": [{"id": "h001"}] * 3, "suggested_top": 3}
        (out / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
        report_highlights(out, out / "highlights.json")

    def events(a):
        hl_file = Path(a.project) / "highlights.json"
        hl = json.loads(hl_file.read_text(encoding="utf-8"))
        hl["suggested_top"] = 1
        hl_file.write_text(json.dumps(hl), encoding="utf-8")

    def assemble(clips_dir, **kw):
        final = clips_dir.parent / "final.mp4"
        final.write_bytes(b"final")
        return final

    monkeypatch.setattr(cutpoint.analyze, "analyze", analyze)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(events_cli, "run_args", events)
    monkeypatch.setattr(edit, "export_clips", lambda project, **kw: project / "clips")
    monkeypatch.setattr(edit, "assemble", assemble)
    assert cli.main(["edit", str(video), "--events", "--no-vision", "--json"]) == 0
    out = capsys.readouterr().out
    doc = _envelope(out)
    assert doc["ok"] is True and doc["command"] == "edit"
    assert doc["outputs"]["suggested_top"] == 1 and doc["outputs"]["candidates"] == 3
    assert out.count('"suggested_top"') == 1  # replaced, not added twice
