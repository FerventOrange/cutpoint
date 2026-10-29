"""``cutpoint extend``: which file a manifest entry is read from; subtitles of the new range."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import media_info, remap, write
from cutpoint import cache, extend, planning, sources, srt
from cutpoint.boundaries import SpeechSettings
from cutpoint.planning import Padding

FMT = {"width": 1920, "height": 1080, "fps": 30, "vertical": False}


@pytest.fixture
def reads(monkeypatch) -> list[Path]:
    """Stub probe and the encoder (no ffmpeg); returns the files clips were encoded from."""
    encoded: list[Path] = []

    def fake_export(src, info, start, end, out, *args):
        encoded.append(src)
        write(out, "encoded")

    monkeypatch.setattr(extend, "probe", media_info)
    monkeypatch.setattr(extend, "export_clip_checked", fake_export)
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))
    return encoded


def _clips_dir(tmp_path: Path, **entry) -> Path:
    clips = tmp_path / "proj" / "clips"
    write(clips / "001_a.mp4")
    e = {"file": "001_a.mp4", "start": 10.0, "end": 20.0} | entry
    write(clips / "manifest.json", json.dumps({"format": FMT, "clips": [e]}))
    return clips


def _run(clips: Path, **kw) -> dict:
    out = extend.extend_clips(clips, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False), **kw)
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]


@pytest.mark.parametrize("case", ["stale-read-from", "remap-wins", "fallback-to-read-from"])
def test_extend_reads_the_source_through_remap_and_read_from_only_as_fallback(
    home, tmp_path, monkeypatch, reads, case
):
    orig, copy, other = tmp_path / "old" / "a.mp4", tmp_path / "new" / "a.mp4", tmp_path / "b.mp4"
    if case == "stale-read-from":  # the copy it read was deleted; the original is back
        write(orig)
        entry = {"source": str(orig), "read_from": str(tmp_path / "gone.mp4")}
        want, recorded = orig, None
    elif case == "remap-wins":  # source_remap reaches the moved footage
        write(copy)
        write(other)
        remap(home, monkeypatch, orig.parent, copy.parent)
        entry = {"source": str(orig), "read_from": str(other)}
        want, recorded = copy, str(copy)
    else:  # neither the original nor a remap target exists; the copy it read does
        write(copy)
        entry = {"source": str(orig), "read_from": str(copy)}
        want, recorded = copy, str(copy)

    new = _run(_clips_dir(tmp_path, **entry))

    assert reads == [want]
    assert new.get("read_from") == recorded


def test_extend_reuse_copies_unchanged_clips_when_only_the_file_read_differs(tmp_path, reads):
    orig = write(tmp_path / "a.mp4")
    prior = tmp_path / "proj" / "earlier"
    write(prior / "001_a.mp4", "earlier")
    entry = {"file": "001_a.mp4", "source": str(orig), "start": 9.0, "end": 21.0}
    entry |= {"read_from": str(tmp_path / "copy.mp4"), "audio_fade": 0.25}
    write(prior / "manifest.json", json.dumps({"format": FMT, "clips": [entry]}))

    out = extend.extend_clips(
        _clips_dir(tmp_path, source=str(orig)),
        Padding(1.0, 1.0),
        speech=SpeechSettings(enabled=False),
        reuse_dir=prior,
    )

    assert reads == []
    assert (out / "001_a.mp4").read_text(encoding="utf-8") == "earlier"


@pytest.mark.parametrize(("entry", "on"), [({}, True), ({"speech_aware": False}, False)])
def test_extend_header_speech_aware_is_off_when_every_clip_turned_it_off(
    tmp_path, monkeypatch, reads, entry, on
):
    monkeypatch.setattr(extend, "speech_pass", lambda *a, **kw: {})
    clips = _clips_dir(tmp_path, source=str(write(tmp_path / "a.mp4")), **entry)
    out = extend.extend_clips(clips, Padding(1.0, 1.0), speech=SpeechSettings())
    doc = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert doc["speech_aware"] == SpeechSettings().to_json() | {"enabled": on}
    assert doc["clips"][0].get("speech_aware", True) is on  # the clip's record as it was


WORDS = [{"start": 9.4, "end": 9.8, "word": "early."}, {"start": 20.2, "end": 20.6, "word": "late"}]


def _srt_clips(tmp_path: Path, srt_file: str | None = "001_a.srt") -> Path:
    """One clip (10-20 s of a.mp4) with an .srt of the old range, listed unless ``srt_file``
    is None."""
    clips = tmp_path / "proj" / "clips"
    write(clips / "001_a.mp4")
    write(clips / "001_a.srt", srt.format_srt([(0.0, 1.0, "stale")]))
    e = {"file": "001_a.mp4", "source": str(write(tmp_path / "a.mp4")), "start": 10.0}
    e |= {"end": 20.0} | ({"srt": srt_file} if srt_file else {})
    write(clips / "manifest.json", json.dumps({"format": FMT, "clips": [e]}))
    return clips


def _manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["clips"][0]


@pytest.mark.parametrize("where", ["in-place", "other-out"])
def test_extend_writes_each_clips_srt_for_its_new_range(tmp_path, monkeypatch, reads, where):
    clips = _srt_clips(tmp_path)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    out = clips if where == "in-place" else tmp_path / "proj" / "ext"
    _run(clips, out_dir=out, force=True)
    # padded by 1 s on each side (9-21 s): both words are in, timed from 9 s
    want = srt.format_srt(srt.clip_cues(WORDS, 9.0, 21.0))
    assert want.startswith("1\n00:00:00,400 --> ")
    assert "late" in want
    assert (out / "001_a.srt").read_text(encoding="utf-8") == want
    assert _manifest(out)["srt"] == "001_a.srt"
    if where == "other-out":  # the input folder is left as it was
        assert "stale" in (clips / "001_a.srt").read_text(encoding="utf-8")


def test_extend_in_place_without_a_transcript_removes_its_old_srt(tmp_path, monkeypatch, reads):
    clips = _srt_clips(tmp_path)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    _run(clips, out_dir=clips, force=True)
    assert not (clips / "001_a.srt").exists()  # listed, so an earlier run's: stale timing
    assert "srt" not in _manifest(clips)


def test_extend_never_overwrites_an_srt_the_manifest_does_not_list(
    tmp_path, monkeypatch, reads, caplog
):
    clips = _srt_clips(tmp_path, srt_file=None)  # 001_a.srt made by hand
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)
    _run(clips, out_dir=clips, force=True)
    assert "stale" in (clips / "001_a.srt").read_text(encoding="utf-8")
    assert "srt" not in _manifest(clips)
    assert any(r.levelname == "WARNING" and "001_a.srt" in r.message for r in caplog.records)


@pytest.mark.parametrize("where", ["in-place", "other-out"])
def test_extend_removes_the_listed_srt_of_a_clip_deleted_during_review(
    tmp_path, monkeypatch, reads, where
):
    clips = _srt_clips(tmp_path)
    out = clips if where == "in-place" else tmp_path / "proj" / "ext"
    src = _manifest(clips)["source"]
    # out holds an earlier run's 001_a and 002_b; 002_b.mp4 was deleted during review
    write(out / "002_b.srt", "old")
    write(out / "notes.srt", "mine")  # not listed: made by hand
    entries = [
        {"file": "001_a.mp4", "source": src, "start": 10.0, "end": 20.0, "srt": "001_a.srt"},
        {"file": "002_b.mp4", "source": src, "start": 30.0, "end": 40.0, "srt": "002_b.srt"},
    ]
    write(out / "manifest.json", json.dumps({"format": FMT, "clips": entries}))
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)

    _run(clips, out_dir=out, force=True)

    assert not (out / "002_b.srt").exists()  # listed by the old manifest, its clip gone
    assert (out / "notes.srt").read_text(encoding="utf-8") == "mine"
    files = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"]
    assert [e["file"] for e in files] == ["001_a.mp4"]
    assert "late" in (out / "001_a.srt").read_text(encoding="utf-8")


def test_extend_out_force_removes_a_listed_clip_it_does_not_re_export_and_its_srt(
    tmp_path, monkeypatch, reads, caplog
):
    """Like clips --force: out's earlier 002_b (deleted from the input during review) is
    listed by out's manifest, so it goes with its .srt instead of staying unlisted."""
    clips = _srt_clips(tmp_path)
    out = tmp_path / "proj" / "ext"
    src = _manifest(clips)["source"]
    for n in ("001_a.mp4", "001_a.srt", "002_b.mp4", "002_b.srt"):
        write(out / n, "old")
    write(out / "009_mine.mp4", "mine")  # not listed: added by hand
    write(out / ".001_x.part.mp4", "partial")  # left by a killed run
    entries = [
        {"file": "001_a.mp4", "source": src, "start": 10.0, "end": 20.0, "srt": "001_a.srt"},
        {"file": "002_b.mp4", "source": src, "start": 30.0, "end": 40.0, "srt": "002_b.srt"},
    ]
    write(out / "manifest.json", json.dumps({"format": FMT, "clips": entries}))
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)

    _run(clips, out_dir=out, force=True)

    assert sorted(p.name for p in out.iterdir()) == [
        "001_a.mp4",
        "001_a.srt",
        "009_mine.mp4",
        "manifest.json",
    ]
    assert (out / "009_mine.mp4").read_text(encoding="utf-8") == "mine"
    files = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"]
    assert [e["file"] for e in files] == ["001_a.mp4"]
    assert "002_b" not in caplog.text
    assert any(  # the kept clip is named in a WARNING
        r.levelname == "WARNING"
        and "009_mine.mp4" in r.message
        and "the old manifest.json does not list them" in r.message
        for r in caplog.records
    )


@pytest.mark.parametrize("where", ["in-place", "other-out"])
def test_extend_without_a_transcript_removes_the_listed_srt_renamed_in_case(
    tmp_path, monkeypatch, reads, where, case_insensitive
):
    """The listed 001_a.srt (in place) or 001_a.mp4 and .srt (in out) were renamed to 001_A
    during review. 001_a.mp4 is written again with no transcript: its old .srt must go, not
    stay unlisted with the old timing."""
    clips = _srt_clips(tmp_path)
    case_insensitive(clips)
    if where == "in-place":
        out = clips
        (clips / "001_a.srt").rename(clips / "001_A.srt")
    else:
        out = tmp_path / "proj" / "ext"
        case_insensitive(out)
        write(out / "001_A.mp4", "old")
        write(out / "001_A.srt", "old")
        write(out / "manifest.json", (clips / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)

    _run(clips, out_dir=out, force=True)

    assert sorted(p.name for p in out.iterdir()) == ["001_a.mp4", "manifest.json"]
    assert (out / "001_a.mp4").read_text(encoding="utf-8") == "encoded"


def test_extend_in_place_keeps_a_clip_renamed_in_case_only_and_its_srt(
    tmp_path, monkeypatch, reads, caplog, case_insensitive
):
    """The manifest lists 002_Joke; review renamed it 002_joke. It has no entry, so it is not
    re-exported, and the listed name must not remove it on a case-insensitive file system."""
    clips = _srt_clips(tmp_path)
    case_insensitive(clips)
    write(clips / "002_joke.mp4", "renamed")
    write(clips / "002_joke.srt", "renamed subs")
    data = json.loads((clips / "manifest.json").read_text(encoding="utf-8"))
    src = data["clips"][0]["source"]
    data["clips"].append(
        {"file": "002_Joke.mp4", "source": src, "start": 30.0, "end": 40.0, "srt": "002_Joke.srt"}
    )
    write(clips / "manifest.json", json.dumps(data))
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: WORDS)

    _run(clips, out_dir=clips, force=True)

    assert (clips / "002_joke.mp4").read_text(encoding="utf-8") == "renamed"
    assert (clips / "002_joke.srt").read_text(encoding="utf-8") == "renamed subs"
    # In place it stays in the folder (and in the edit): a Kept warning, not "skipped".
    assert f"Kept 002_joke.mp4 in {clips.absolute()}: the old manifest.json" in caplog.text
    assert "skipped" not in caplog.text


@pytest.mark.parametrize(("language", "word"), [(None, "hello"), ("de", "hallo")])
def test_extend_subtitles_from_the_transcript_of_the_analysis_language(
    home, tmp_path, reads, language, word
):
    """#371: the project's highlights.json settings.language picks the cached transcript."""
    clips = _srt_clips(tmp_path)
    src = tmp_path / "a.mp4"
    for name, text in (("transcript-small.json", "hello"), ("transcript-small-de.json", "hallo")):
        segment = {"start": 12.0, "end": 12.4, "text": text,
                   "words": [{"start": 12.0, "end": 12.4, "word": text}]}  # fmt: skip
        cache.save_json(cache.source_dir(src) / name, [segment])
    write(clips.parent / "highlights.json", json.dumps({"settings": {"language": language}}))
    _run(clips, out_dir=tmp_path / "proj" / "ext")
    assert word in (tmp_path / "proj" / "ext" / "001_a.srt").read_text(encoding="utf-8")
