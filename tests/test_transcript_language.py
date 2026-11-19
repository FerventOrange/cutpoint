"""The transcript cache is keyed by the forced language (``analyze --language``) (#371)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import transcribed_both, write
from cutpoint import analyze, cache, sources, srt
from cutpoint.transcribe import transcript_name


def _whisper(calls: list[str], text: str):
    def transcribe(wav, on_progress=None):
        calls.append(text)
        return [{"start": 0.0, "end": 1.0, "text": text, "words": []}]

    return transcribe


def test_a_forced_language_re_transcribes_past_a_detected_language_cache(tmp_path):
    cache.save_json(tmp_path / "transcript-m.json", [{"start": 0, "end": 1, "text": "hello"}])
    calls: list[str] = []
    tr = SimpleNamespace(model_name="m", language="de", transcribe=_whisper(calls, "hallo"))
    got = analyze._transcript(tmp_path / "a.wav", tmp_path, tr)
    assert calls == ["hallo"]  # Whisper ran: the English cache was not used
    assert got[0]["text"] == "hallo"
    assert cache.load_json(tmp_path / "transcript-m-de.json") == got
    assert cache.load_json(tmp_path / "transcript-m.json")[0]["text"] == "hello"  # untouched
    analyze._transcript(tmp_path / "a.wav", tmp_path, tr)
    assert calls == ["hallo"]  # its own cache hits now


def test_no_language_still_hits_the_old_cache_name(tmp_path):
    cache.save_json(tmp_path / "transcript-m.json", [{"start": 0, "end": 1, "text": "hello"}])
    cache.save_json(tmp_path / "transcript-m-de.json", [{"start": 0, "end": 1, "text": "hallo"}])
    calls: list[str] = []
    tr = SimpleNamespace(model_name="m", language=None, transcribe=_whisper(calls, "x"))
    assert analyze._transcript(tmp_path / "a.wav", tmp_path, tr)[0]["text"] == "hello"
    assert calls == []


@pytest.mark.parametrize(
    ("files", "language", "want"),
    [
        (["transcript-small.json", "transcript-small-de.json"], "de", "transcript-small-de.json"),
        (["transcript-small-de.json", "transcript-large-v3-de.json"], "de",
         "transcript-large-v3-de.json"),  # most accurate model first
        (["transcript-base.json", "transcript-small-de.json"], None, "transcript-base.json"),
        (["transcript-small-de.json"], None, "transcript-small-de.json"),  # the only one
        (["transcript-small.json"], "de", "transcript-small.json"),  # analysed before #371
    ],
)  # fmt: skip
def test_transcript_file_picks_the_language_of_the_analysis(tmp_path, files, language, want):
    for name in files:
        cache.save_json(tmp_path / name, [])
    assert sources._transcript_file(tmp_path, language) == tmp_path / want


def test_source_words_and_srt_lookup_read_the_given_language(home, tmp_path):
    src = transcribed_both(tmp_path / "a.mp4")
    assert [w["word"] for w in sources.source_words(src)] == ["hello"]
    assert [w["word"] for w in sources.source_words(src, "de")] == ["hallo"]
    assert srt.words_lookup()(str(src))[0]["word"] == "hello"
    assert srt.words_lookup("de")(str(src))[0]["word"] == "hallo"
    subs = srt.ClipSubtitles(language="de")
    subs.write(tmp_path / "001_a.mp4", str(src), 100.0, 104.0)
    assert "hallo" in (tmp_path / "001_a.srt").read_text(encoding="utf-8")


def test_project_language_is_none_without_a_usable_highlights_file(tmp_path):
    """A project without highlights.json, or an old list-shaped one, has no language."""
    assert sources.project_language(tmp_path) is None
    write(tmp_path / "highlights.json", json.dumps([]))
    assert sources.project_language(tmp_path) is None


def test_language_case_is_ignored(tmp_path: Path) -> None:
    assert transcript_name("m", "DE") == "transcript-m-de.json"
    (tmp_path / "transcript-large-v3.json").write_text("[]")
    (tmp_path / "transcript-large-v3-de.json").write_text("[]")
    assert sources._transcript_file(tmp_path, "DE") == tmp_path / "transcript-large-v3-de.json"
