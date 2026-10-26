"""SRT subtitles next to exported clips and assembled videos (#5)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, cache, clips, sources, srt
from cutpoint.ffmpeg_utils import MediaInfo


def _words(*items: tuple[float, float, str]) -> list[dict]:
    return [{"start": s, "end": e, "word": w} for s, e, w in items]


def _info(path: Path, duration: float) -> MediaInfo:
    return media_info(path, duration)


# Source "a.mp4": "hello there." at 101 s, "bye" at 104.5 s (past a 4 s clip from 100 s).
WORDS = _words((101.0, 101.4, "hello"), (101.5, 101.9, "there."), (104.5, 104.9, "bye"))


def test_single_clip_srt_is_timed_to_the_clip_start():
    cues = srt.clip_cues(WORDS, 100.0, 104.0)
    assert srt.format_srt(cues) == "1\n00:00:01,000 --> 00:00:02,050\nhello there.\n\n"
    # cue times clipped to the clip; hours, carries and rounding
    assert srt.srt_time(3723.4567) == "01:02:03,457"
    assert srt.srt_time(59.9996) == "00:01:00,000"


def test_extended_clip_shifts_by_its_padded_start():
    """A clip padded (extend) by 1.5 s starts earlier in the source: its subtitles start later."""
    plain = srt.clip_cues(WORDS, 100.0, 104.0)
    padded = srt.clip_cues(WORDS, 98.5, 105.5)
    assert padded[0][0] == pytest.approx(plain[0][0] + 1.5)
    assert padded[-1][2] == "bye"  # the padded end now covers it


@pytest.mark.parametrize("fade", [0.0, 0.5], ids=["cut", "fade"])
def test_final_cues_follow_cuts_and_crossfade_overlap(tmp_path, fade):
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    infos = [_info(files[0], 4.0), _info(files[1], 3.0)]
    entries = {
        "001_a.mp4": {"source": "a.mp4", "start": 100.0, "end": 104.0},
        "002_b.mp4": {"source": "b.mp4", "start": 10.0, "end": 13.0},
    }
    by_source = {
        "a.mp4": _words((103.5, 104.0, "late")),  # held past the clip end, into the overlap
        "b.mp4": _words((10.2, 10.6, "first"), (12.0, 12.4, "second")),
    }
    cues, missing, _ = srt.final_cues(files, infos, entries, fade, by_source.get)
    assert missing == []
    second = 4.0 - fade  # where clip 2 starts in the final
    assert [c[2] for c in cues] == ["late", "first", "second"]
    assert cues[1][0] == pytest.approx(second + 0.2)
    assert cues[2][0] == pytest.approx(second + 2.0)
    assert cues[0][1] <= cues[1][0]  # never overlapping, also inside a crossfade
    assert assemble.clip_offsets(infos, fade) == [0.0, second]


def test_missing_transcript_writes_no_srt_and_says_so_once(tmp_path, monkeypatch, caplog):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    files = [clips_dir / "001_a.mp4", clips_dir / "002_b.mp4"]  # hand-made: no manifest
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    caplog.set_level(logging.INFO)
    out = tmp_path / "final.mp4"
    infos = [_info(f, 2.0) for f in files]
    assert srt.write_final_srt(clips_dir, files, infos, 0.0, out) is None
    assert not out.with_suffix(".srt").exists()
    assert len([r for r in caplog.records if "No subtitles" in r.message]) == 1


def _project(home: Path) -> tuple[Path, Path, Path]:
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    src = project / "src.mp4"
    src.write_bytes(b"x")
    seg = project / "seg.json"
    seg.write_text(
        json.dumps([{"source": str(src), "start": 100.0, "end": 104.0, "label": "a",
                     "speech_aware": False, "smart_tail": False}]),
        encoding="utf-8",
    )  # fmt: skip
    return project, seg, project / "clips"


def _fake_export(src, info, start, end, out, *_a, **_kw) -> None:
    out.write_bytes(b"new")


def test_clips_write_srt_and_force_replaces_only_their_own(home, monkeypatch, caplog):
    project, seg, out = _project(home)
    monkeypatch.setattr(clips, "probe", lambda p: _info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    clips.export_clips(project, segments=seg)
    assert (out / "001_a.srt").read_text(encoding="utf-8").startswith("1\n00:00:01,000 --> ")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clips"][0]["srt"] == "001_a.srt"

    # --force: the old run's .srt goes even when the new run has no transcript for the
    # clip; an .srt the old manifest does not list (made by hand) stays.
    (out / "notes.srt").write_text("mine", encoding="utf-8")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    caplog.set_level(logging.INFO)
    clips.export_clips(project, segments=seg, force=True)
    assert not (out / "001_a.srt").exists()
    assert (out / "notes.srt").read_text(encoding="utf-8") == "mine"
    assert "srt" not in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    assert "No subtitles for 1 of 1 clips" in caplog.text

    # an .srt under the new clip's name that the old manifest does not list is not overwritten
    (out / "001_a.srt").write_text("mine too", encoding="utf-8")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    caplog.clear()
    clips.export_clips(project, segments=seg, force=True)
    assert (out / "001_a.srt").read_text(encoding="utf-8") == "mine too"
    assert "srt" not in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    assert any(r.levelname == "WARNING" and "001_a.srt" in r.message for r in caplog.records)


def test_listed_files_are_the_bare_clip_and_srt_names_of_the_manifest(tmp_path):
    out = tmp_path / "clips"
    out.mkdir()
    entries = [
        {"file": "001_a.mp4", "srt": "001_a.srt"},
        {"file": "002_b.mp4"},  # no .srt listed: 002_b.srt is not cutpoint's
        {"file": "003_c.mp4", "srt": "../x.srt"},  # never a path out of the folder
        {"file": "notes.txt", "srt": "004_d.mp4"},  # never another kind of file
    ]
    (out / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    assert srt.listed_files(out) == ["001_a.mp4", "001_a.srt", "002_b.mp4", "003_c.mp4"]
    (out / "manifest.json").unlink()
    assert srt.listed_files(out) is None  # no manifest: nothing is cutpoint's


def test_clips_force_removes_the_listed_srt_of_a_clip_deleted_during_review(home, monkeypatch):
    """Like an in-place extend (#239): the new manifest no longer lists it."""
    project, seg, out = _project(home)
    monkeypatch.setattr(clips, "probe", lambda p: _info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    clips.export_clips(project, segments=seg)
    (out / "001_a.mp4").unlink()  # rejected during review
    pick = json.loads(seg.read_text(encoding="utf-8"))[0] | {"label": "b"}
    seg.write_text(json.dumps([pick]), encoding="utf-8")
    clips.export_clips(project, segments=seg, force=True)
    assert not (out / "001_a.srt").exists()
    assert (out / "001_b.srt").is_file()


def test_clips_force_keeps_a_clip_renamed_in_case_only_and_its_srt(home, monkeypatch, caplog):
    """The manifest lists 001_Joke; review renamed it 001_joke. The listed name must not
    remove it on a case-insensitive file system: it stays, with the Kept warning."""
    project, seg, out = _project(home)
    monkeypatch.setattr(clips, "probe", lambda p: _info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    clips.export_clips(project, segments=seg)
    if not (out / "001_A.mp4").exists():
        pytest.skip("case-sensitive file system")
    data = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    data["clips"][0] |= {"file": "001_Joke.mp4", "srt": "001_Joke.srt"}
    (out / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (out / "001_a.mp4").rename(out / "001_joke.mp4")
    (out / "001_a.srt").rename(out / "001_joke.srt")
    pick = json.loads(seg.read_text(encoding="utf-8"))[0] | {"label": "b"}
    seg.write_text(json.dumps([pick]), encoding="utf-8")
    clips.export_clips(project, segments=seg, force=True)
    assert (out / "001_joke.mp4").read_bytes() == b"new"  # the first run's clip, kept
    assert (out / "001_joke.srt").is_file()
    assert (out / "001_b.mp4").is_file()
    assert any(
        r.levelname == "WARNING" and "Kept 001_joke.mp4" in r.message for r in caplog.records
    )


@pytest.mark.parametrize("renamed", [("001_a.mp4", "001_a.srt"), ("001_a.srt",)])
def test_clips_force_replaces_the_listed_srt_of_a_replaced_clip_renamed_in_case(
    home, monkeypatch, renamed
):
    """Review renamed 001_a (or only its .srt) to 001_A. The run writes 001_a again without
    a transcript: the clip is replaced, so its old .srt goes too, not stays with stale cues."""
    project, seg, out = _project(home)
    monkeypatch.setattr(clips, "probe", lambda p: _info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    clips.export_clips(project, segments=seg)
    if not (out / "001_A.mp4").exists():
        pytest.skip("case-sensitive file system")
    for n in renamed:
        (out / n).rename(out / n.replace("_a", "_A"))
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    clips.export_clips(project, segments=seg, force=True)
    assert sorted(p.name for p in out.iterdir()) == ["001_a.mp4", "manifest.json"]


def test_renamed_clip_uses_its_own_srt_in_the_final(tmp_path):
    """Reordering by renaming (005_x -> 001b_x) loses the manifest entry: the clip's own
    .srt, renamed with it, is placed at its offset and clipped to it."""
    files = [tmp_path / "001_a.mp4", tmp_path / "001b_x.mp4"]
    infos = [_info(files[0], 4.0), _info(files[1], 3.0)]
    srt.write_srt(tmp_path / "001b_x.srt", [(0.5, 1.5, "moved"), (2.5, 9.0, "long")])
    entries = {"001_a.mp4": {"source": "a.mp4", "start": 100.0, "end": 104.0}}
    cues, missing, _ = srt.final_cues(files, infos, entries, 0.0, {"a.mp4": WORDS}.get)
    assert missing == []
    assert cues[-2:] == [(4.5, 5.5, "moved"), (6.5, 7.0, "long")]


def _lavfi_clip(out: Path, seconds: float, audio: float | None = None) -> None:
    """A 64x48 clip of ``seconds``; its audio runs ``audio`` s (default: as long)."""
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=size=64x48:rate=10:duration={seconds}", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={audio or seconds}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", *([] if audio else ["-shortest"]), str(out)],
        check=True,
    )  # fmt: skip


@needs_ffmpeg
def test_assemble_writes_final_srt_on_the_crossfaded_timeline(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("001_a.mp4", "002_b.mp4"):
        _lavfi_clip(clips_dir / name, 2.0)
    (clips_dir / "manifest.json").write_text(
        json.dumps({"clips": [
            {"file": "001_a.mp4", "source": "a.mp4", "start": 0.0, "end": 2.0},
            {"file": "002_b.mp4", "source": "b.mp4", "start": 50.0, "end": 52.0},
        ]}),
        encoding="utf-8",
    )  # fmt: skip
    by_source = {"a.mp4": _words((0.2, 0.6, "one")), "b.mp4": _words((51.0, 51.4, "two"))}
    monkeypatch.setattr(sources, "source_words", lambda src, _lang=None: by_source.get(src.name))
    final = assemble.assemble(clips_dir, transition="fade", fade_dur=0.5)
    text = (tmp_path / "final.srt").read_text(encoding="utf-8")
    assert final == tmp_path / "final.mp4"
    # clip 2 starts at its first clip's video length minus the 0.5 s crossfade
    second = assemble.probe(clips_dir / "001_a.mp4").vduration - 0.5
    assert f"{srt.srt_time(second + 1.0)} --> " in text
    assert text.startswith("1\n00:00:00,200 --> ")


@needs_ffmpeg
@pytest.mark.parametrize("baked", [0.25, 0.0], ids=["stream-copy", "audio-faded"])
def test_assemble_cut_srt_follows_where_concat_starts_each_clip(tmp_path, monkeypatch, baked):
    """Clips whose audio runs 0.1 s past their 2 s of video: concat starts each next clip
    2.1 s on, and final.srt follows it instead of drifting 0.1 s per clip."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    names = ["001_a.mp4", "002_b.mp4", "003_c.mp4"]
    for name in names:
        _lavfi_clip(clips_dir / name, 2.0, audio=2.1)
    entries = [
        {"file": n, "source": f"{n[4]}.mp4", "start": 0.0, "end": 2.0, "audio_fade": baked}
        for n in names
    ]
    (clips_dir / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    words = _words((1.0, 1.4, "word"))
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: words)
    assemble.assemble(clips_dir)
    keyframes = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(tmp_path / "final.mp4")],
        capture_output=True, text=True, check=True,
    ).stdout.replace(",", " ").split()  # fmt: skip
    starts = [float(k) - float(keyframes[0]) for k in keyframes]  # each clip opens on one
    assert starts == pytest.approx([0.0, 2.1, 4.2], abs=0.03)
    cues = srt.read_srt(tmp_path / "final.srt")
    assert [c[0] for c in cues] == pytest.approx([s + 1.0 for s in starts], abs=0.01)


def _two_language_projects(home: Path, **seg: Any) -> dict[str | None, Path]:
    """One source with a cached detected-language ("hello") and German ("hallo")
    transcript, and a project of it per analysis language, each with seg.json (its one
    segment's tails and speech pass off unless ``seg`` turns them on)."""
    src = home / "footage" / "src.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    for name, word in (("transcript-small.json", "hello"), ("transcript-small-de.json", "hallo")):
        segment = {"start": 101.0, "end": 101.4, "text": word,
                   "words": [{"start": 101.0, "end": 101.4, "word": word}]}  # fmt: skip
        cache.save_json(cache.source_dir(src) / name, [segment])
    projects = {}
    for language in (None, "de"):
        project = home / "projects" / (language or "auto")
        project.mkdir(parents=True)
        hl = {"style": "gaming", "settings": {"language": language}, "sources": []}
        (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
        entry = {"source": str(src), "start": 100.0, "end": 104.0, "label": "a",
                 "speech_aware": False, "smart_tail": False} | seg  # fmt: skip
        (project / "seg.json").write_text(json.dumps([entry]), encoding="utf-8")
        projects[language] = project
    return projects


def _export(project: Path, monkeypatch, **kw: Any) -> Path:
    monkeypatch.setattr(clips, "probe", lambda p: _info(p, 200.0))
    monkeypatch.setattr(clips, "export_clip_checked", _fake_export)
    clips.export_clips(project, segments=project / "seg.json", **kw)
    return project / "clips"


def test_clips_srt_uses_the_transcript_of_the_analysis_language(home, monkeypatch):
    """#371: a project analysed with --language de subtitles from transcript-small-de.json;
    an auto-detect project of the same source keeps transcript-small.json."""
    projects = _two_language_projects(home)
    for language, word in ((None, "hello"), ("de", "hallo")):
        out = _export(projects[language], monkeypatch)
        assert word in (out / "001_a.srt").read_text(encoding="utf-8")


@needs_ffmpeg
def test_final_srt_uses_the_transcript_of_the_analysis_language(home, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    projects = _two_language_projects(home)
    for language, word in ((None, "hello"), ("de", "hallo")):
        out = _export(projects[language], monkeypatch)
        (out / "001_a.mp4").unlink()  # a real clip in place of the fake export
        _lavfi_clip(out / "001_a.mp4", 4.0)
        assemble.assemble(out)
        assert word in (projects[language] / "final.srt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("seg", "kw"),
    [
        ({"smart_tail": True}, {"smart_tail_on": True}),  # the smart tail reads the words
        ({"smart_tail": True}, {"smart_tail_on": False, "event_tails": True}),  # event tail
        ({"speech_aware": True}, {"smart_tail_on": False}),  # the speech pass
    ],
    ids=["smart-tail", "event-tail", "speech-pass"],
)
def test_clips_planning_reads_the_transcript_of_the_analysis_language(home, monkeypatch, seg, kw):
    """#371: the words for the tails and the speech pass come from transcript-small-de.json
    too, not only the subtitles."""
    projects = _two_language_projects(home, **seg)
    project = projects["de"]
    if kw.get("event_tails"):
        (project / "events.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    langs = []
    real = sources.source_words

    def spy(src, language=None):
        langs.append(language)
        return real(src, language)

    monkeypatch.setattr(sources, "source_words", spy)
    _export(project, monkeypatch, **kw)
    assert langs == ["de", "de"]  # one read to plan the clip, one for its subtitles
